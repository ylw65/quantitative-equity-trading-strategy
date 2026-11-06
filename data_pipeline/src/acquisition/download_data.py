"""Resilient, resumable Yahoo Finance acquisition.

The acquisition layer intentionally preserves vendor observations. Validation,
return diagnostics, and research-facing standardization belong to later stages.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import logging
import os
import platform
import re
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd
import yfinance as yf
import yfinance.shared as yf_shared

from src.common import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    atomic_write_json,
    configure_logging,
    ensure_parent,
    load_config,
    project_path,
)


LOGGER = logging.getLogger("us_equity_data.acquisition")
Downloader = Callable[..., pd.DataFrame]

REQUIRED_VENDOR_FIELDS = {
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "dividends",
    "stock_splits",
}
PRICE_VALUE_FIELDS = {"open", "high", "low", "close", "adj_close"}


@dataclass(frozen=True)
class SecurityRequest:
    """One downloadable equity security/ticker mapping."""

    security_id: str
    ticker_2023: str
    yahoo_ticker: str
    company_name: str = ""
    exchange: str = ""
    current_ticker: str = ""


def utc_now() -> str:
    """Return a timezone-aware UTC ISO timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def elapsed_start_timestamp(started_monotonic: float) -> str:
    """Derive an ISO wall-clock start time from a monotonic timer value."""
    started = datetime.now(timezone.utc) - pd.Timedelta(
        seconds=time.monotonic() - started_monotonic
    )
    return started.isoformat(timespec="seconds")


def _clean_string(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _normalise_column(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    aliases = {
        "adj_close": "adj_close",
        "adjclose": "adj_close",
        "stock_splits": "stock_splits",
        "stock_split": "stock_splits",
        "capital_gains": "capital_gains",
    }
    return aliases.get(text, text)


def load_requests(
    universe_path: str | Path,
    limit: int | None = None,
    explicit_tickers: Iterable[str] | None = None,
) -> list[SecurityRequest]:
    """Load included securities or create a clearly labelled smoke-test list.

    Explicit tickers are supported for source validation before the historical
    universe is complete. They are not a substitute for the research universe.
    """
    tickers = [
        item.strip().upper()
        for item in (explicit_tickers or [])
        if item and item.strip()
    ]
    if tickers:
        requests = [
            SecurityRequest(
                security_id=f"TICKER2023:{ticker}",
                ticker_2023=ticker,
                yahoo_ticker=ticker,
                company_name="",
                exchange="",
                    current_ticker="",
            )
            for ticker in dict.fromkeys(tickers)
        ]
    else:
        path = project_path(universe_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Security master not found at {path}. Build the historical "
                "universe first or use --tickers for a smoke test."
            )
        table = (
            pd.read_parquet(path)
            if path.suffix.lower() == ".parquet"
            else pd.read_csv(path, low_memory=False)
        )
        if "included" in table.columns:
            included = table["included"]
            if included.dtype == object:
                included = included.astype(str).str.lower().isin(
                    {"true", "1", "yes", "y"}
                )
            table = table[included.fillna(False)]

        yahoo_col = (
            "yahoo_ticker"
            if "yahoo_ticker" in table.columns
            else "ticker"
        )
        ticker_col = (
            "ticker_2023"
            if "ticker_2023" in table.columns
            else yahoo_col
        )
        if yahoo_col not in table.columns:
            raise ValueError("Security master lacks yahoo_ticker/ticker.")

        requests = []
        for _, row in table.iterrows():
            yahoo_ticker = _clean_string(row.get(yahoo_col)).upper()
            ticker = _clean_string(row.get(ticker_col)).upper() or yahoo_ticker
            if not yahoo_ticker:
                continue
            cik = _clean_string(row.get("cik"))
            security_id = _clean_string(row.get("security_id"))
            if not security_id:
                security_id = (
                    f"SEC:CIK{int(float(cik)):010d}:{ticker}"
                    if cik
                    else f"TICKER2023:{ticker}"
                )
            requests.append(
                SecurityRequest(
                    security_id=security_id,
                    ticker_2023=ticker,
                    yahoo_ticker=yahoo_ticker,
                    company_name=_clean_string(
                        row.get("company_name", row.get("name", ""))
                    ),
                    exchange=_clean_string(row.get("exchange", "")),
                    current_ticker=_clean_string(row.get("current_ticker", "")).upper(),
                )
            )

    deduplicated: dict[str, SecurityRequest] = {}
    for request in requests:
        if request.yahoo_ticker in deduplicated:
            raise ValueError(
                "Duplicate Yahoo ticker mapping in security master: "
                f"{request.yahoo_ticker}"
            )
        deduplicated[request.yahoo_ticker] = request
    result = list(deduplicated.values())
    return result[:limit] if limit is not None else result


def _extract_ticker_frame(
    data: pd.DataFrame | None, ticker: str, batch_size: int
) -> pd.DataFrame:
    """Extract a ticker frame from yfinance's variable column layout."""
    if data is None or len(data) == 0:
        return pd.DataFrame()
    if not isinstance(data.columns, pd.MultiIndex):
        return data.copy() if batch_size == 1 else pd.DataFrame()

    for level in range(data.columns.nlevels):
        labels = {str(value).upper() for value in data.columns.get_level_values(level)}
        if ticker.upper() in labels:
            frame = data.xs(ticker.upper(), axis=1, level=level, drop_level=True)
            if isinstance(frame, pd.Series):
                frame = frame.to_frame(name="close")
            return frame.copy()
    return pd.DataFrame()


def normalize_vendor_frame(
    frame: pd.DataFrame,
    request: SecurityRequest,
    start_date: str,
    end_date_exclusive: str,
    download_id: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Create a raw audit frame without filling genuine observations.

    yfinance sometimes returns a full date index populated entirely with NaN
    for a missing/delisted ticker. Those are placeholder rows, not price
    observations, and are removed before success/failure classification.
    """
    if frame is None or len(frame) == 0:
        return pd.DataFrame(), sorted(REQUIRED_VENDOR_FIELDS)

    result = frame.copy()
    if "date" not in {_normalise_column(c) for c in result.columns}:
        result = result.reset_index()

    source_columns = list(result.columns)
    normalised: dict[Any, str] = {}
    used: set[str] = set()
    for column in source_columns:
        name = _normalise_column(column)
        if not name or name in used:
            base = name or "unnamed"
            suffix = 2
            name = f"{base}_{suffix}"
            while name in used:
                suffix += 1
                name = f"{base}_{suffix}"
        normalised[column] = name
        used.add(name)
    result = result.rename(columns=normalised)

    if "date" not in result.columns:
        raise ValueError("Unable to identify a date index or column.")
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result = result.loc[
        result["date"].notna()
        & (result["date"] >= pd.Timestamp(start_date))
        & (result["date"] < pd.Timestamp(end_date_exclusive))
    ].copy()
    result = result.sort_values("date", kind="stable")

    missing_fields = sorted(REQUIRED_VENDOR_FIELDS - set(result.columns))
    for field in REQUIRED_VENDOR_FIELDS.intersection(result.columns):
        result[field] = pd.to_numeric(result[field], errors="coerce")

    price_columns = sorted(PRICE_VALUE_FIELDS.intersection(result.columns))
    if not price_columns:
        return pd.DataFrame(), missing_fields
    result = result.loc[result[price_columns].notna().any(axis=1)].copy()
    if result.empty:
        return pd.DataFrame(), missing_fields

    result.insert(1, "ticker", request.yahoo_ticker)
    result.insert(2, "security_id", request.security_id)
    result.insert(3, "ticker_2023", request.ticker_2023)
    result.insert(4, "provider", "yahoo_finance")
    result.insert(5, "download_id", download_id)
    result.insert(6, "retrieved_at_utc", utc_now())
    return result, missing_fields


def raw_output_path(raw_dir: Path, request: SecurityRequest) -> Path:
    """Return a stable, filesystem-safe per-security raw checkpoint path."""
    safe_ticker = re.sub(r"[^A-Za-z0-9._-]+", "_", request.yahoo_ticker)
    key = hashlib.sha256(
        f"{request.security_id}|{request.yahoo_ticker}".encode("utf-8")
    ).hexdigest()[:10]
    return raw_dir / f"{safe_ticker}__{key}.parquet"


def manifest_path_value(path: Path) -> str:
    """Store project-relative paths when possible and absolute paths otherwise."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Atomically write Parquet, replacing only this security's checkpoint."""
    target = ensure_parent(path)
    with tempfile.NamedTemporaryFile(
        suffix=".parquet", dir=target.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False, engine="pyarrow")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Atomically write a manifest or issue table."""
    target = ensure_parent(path)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", encoding="utf-8", dir=target.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def yahoo_download(
    tickers: list[str],
    start: str,
    end: str,
    timeout: int,
) -> pd.DataFrame:
    """Thin yfinance wrapper with the required acquisition flags."""
    try:
        yf_shared._ERRORS.clear()
    except AttributeError:
        pass
    return yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        actions=True,
        repair=True,
        group_by="ticker",
        threads=False,
        progress=False,
        timeout=timeout,
    )


def vendor_errors(tickers: Iterable[str]) -> dict[str, str]:
    """Return yfinance's per-ticker error map when the provider supplies one."""
    errors = getattr(yf_shared, "_ERRORS", {}) or {}
    wanted = {ticker.upper() for ticker in tickers}
    return {
        str(ticker).upper(): f"{type(error).__name__}: {error}"
        for ticker, error in errors.items()
        if str(ticker).upper() in wanted
    }


def _is_permanent_unavailable_error(message: str) -> bool:
    """Identify Yahoo responses that retries cannot plausibly resolve."""
    text = str(message or "").lower()
    return any(
        marker in text
        for marker in (
            "yftzmissingerror",
            "yfpricesmissingerror",
            "possibly delisted",
            "no timezone found",
            "no price data found",
            "quote not found",
            "404",
        )
    )


def _is_permanent_request_unavailable(
    request: SecurityRequest, message: str
) -> bool:
    """Keep current listings retryable after transport/timezone failures."""
    return not request.current_ticker and _is_permanent_unavailable_error(message)


def _package_versions() -> dict[str, str]:
    names = [
        "pandas",
        "numpy",
        "yfinance",
        "pyarrow",
        "curl_cffi",
    ]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return versions


def write_acquisition_provenance(
    config: dict[str, Any],
    metadata_dir: Path,
    start_date: str,
    end_date_exclusive: str,
) -> Path:
    """Record reproducible acquisition parameters and runtime versions."""
    payload = {
        "written_at_utc": utc_now(),
        "provider": "yahoo_finance",
        "library": "yfinance",
        "package_versions": _package_versions(),
        "python": sys.version,
        "platform": platform.platform(),
        "start_date": start_date,
        "end_date_exclusive": end_date_exclusive,
        "download_parameters": config["download"],
        "vendor_fields_retained": sorted(REQUIRED_VENDOR_FIELDS),
        "research_constraints": config["research_constraints"],
    }
    path = metadata_dir / "acquisition_run_provenance.json"
    atomic_write_json(payload, path)
    return path


class AcquisitionJournal:
    """Maintain a latest-result manifest and a failure/partial register."""

    COLUMNS = [
        "security_id",
        "ticker_2023",
        "yahoo_ticker",
        "company_name",
        "exchange",
        "download_status",
        "history_status",
        "batch_number",
        "attempts",
        "n_observations",
        "first_date",
        "last_date",
        "missing_vendor_fields",
        "raw_path",
        "download_id",
        "attempted_at_utc",
        "completed_at_utc",
        "duration_seconds",
        "error",
    ]

    def __init__(self, manifest_path: Path, failures_path: Path) -> None:
        self.manifest_path = manifest_path
        self.failures_path = failures_path
        if manifest_path.exists():
            self.manifest = pd.read_csv(manifest_path, low_memory=False)
        else:
            self.manifest = pd.DataFrame(columns=self.COLUMNS)
        self._records: list[dict[str, Any]] = self.manifest.to_dict("records")

    def get(self, yahoo_ticker: str) -> dict[str, Any] | None:
        matches = [
            row for row in self._records if row["yahoo_ticker"] == yahoo_ticker
        ]
        return matches[-1] if matches else None

    def record(self, result: dict[str, Any]) -> None:
        ticker = result["yahoo_ticker"]
        self._records = [
            row for row in self._records if row["yahoo_ticker"] != ticker
        ]
        self._records.append({column: result.get(column) for column in self.COLUMNS})

    def flush(self) -> None:
        self.manifest = pd.DataFrame(self._records, columns=self.COLUMNS)
        atomic_write_csv(self.manifest, self.manifest_path)
        issues = self.manifest[
            self.manifest["download_status"].isin(["failed", "partial"])
        ].copy()
        atomic_write_csv(issues, self.failures_path)


def recover_manifest_from_raw(
    request: SecurityRequest, raw_path: Path, download_id: str
) -> dict[str, Any] | None:
    """Recover a completed checkpoint that predates the last manifest flush."""
    if not raw_path.exists():
        return None
    try:
        frame = pd.read_parquet(raw_path)
    except Exception:
        return None
    if (
        frame.empty
        or "date" not in frame.columns
        or not frame[list(PRICE_VALUE_FIELDS.intersection(frame.columns))]
        .notna()
        .any(axis=None)
    ):
        return None
    first = pd.to_datetime(frame["date"], errors="coerce").min()
    last = pd.to_datetime(frame["date"], errors="coerce").max()
    missing = sorted(REQUIRED_VENDOR_FIELDS - set(frame.columns))
    return {
        "security_id": request.security_id,
        "ticker_2023": request.ticker_2023,
        "yahoo_ticker": request.yahoo_ticker,
        "company_name": request.company_name,
        "exchange": request.exchange,
        "download_status": "partial" if missing else "success",
        "history_status": "checkpoint_recovered",
        "batch_number": "",
        "attempts": "",
        "n_observations": len(frame),
        "first_date": first.date().isoformat() if pd.notna(first) else "",
        "last_date": last.date().isoformat() if pd.notna(last) else "",
        "missing_vendor_fields": "|".join(missing),
        "raw_path": manifest_path_value(raw_path),
        "download_id": download_id,
        "attempted_at_utc": "",
        "completed_at_utc": utc_now(),
        "duration_seconds": "",
        "error": "",
    }


def _history_status(
    frame: pd.DataFrame, start_date: str, end_date_exclusive: str
) -> str:
    if frame.empty:
        return "unavailable"
    first = pd.to_datetime(frame["date"]).min()
    last = pd.to_datetime(frame["date"]).max()
    requested_start = pd.Timestamp(start_date)
    requested_end = pd.Timestamp(end_date_exclusive) - pd.Timedelta(days=1)
    first_gap = (first - requested_start).days
    last_gap = (requested_end - last).days
    if first_gap <= 7 and last_gap <= 7:
        return "observed_full_requested_period"
    if first_gap > 7 and last_gap > 7:
        return "observed_middle_subperiod"
    if first_gap > 7:
        return "starts_after_requested_start"
    return "ends_before_requested_end"


def _record_success(
    request: SecurityRequest,
    raw_frame: pd.DataFrame,
    missing_fields: list[str],
    raw_path: Path,
    journal: AcquisitionJournal,
    download_id: str,
    batch_number: int,
    attempts: int,
    started: float,
    start_date: str,
    end_date_exclusive: str,
) -> None:
    first = pd.to_datetime(raw_frame["date"]).min()
    last = pd.to_datetime(raw_frame["date"]).max()
    journal.record(
        {
            "security_id": request.security_id,
            "ticker_2023": request.ticker_2023,
            "yahoo_ticker": request.yahoo_ticker,
            "company_name": request.company_name,
            "exchange": request.exchange,
            "download_status": "partial" if missing_fields else "success",
            "history_status": _history_status(
                raw_frame, start_date, end_date_exclusive
            ),
            "batch_number": batch_number,
            "attempts": attempts,
            "n_observations": len(raw_frame),
            "first_date": first.date().isoformat(),
            "last_date": last.date().isoformat(),
            "missing_vendor_fields": "|".join(missing_fields),
            "raw_path": manifest_path_value(raw_path),
            "download_id": download_id,
            "attempted_at_utc": elapsed_start_timestamp(started),
            "completed_at_utc": utc_now(),
            "duration_seconds": round(time.monotonic() - started, 3),
            "error": "",
        }
    )


def _record_failure(
    request: SecurityRequest,
    journal: AcquisitionJournal,
    download_id: str,
    batch_number: int,
    attempts: int,
    started: float,
    error: str,
) -> None:
    journal.record(
        {
            "security_id": request.security_id,
            "ticker_2023": request.ticker_2023,
            "yahoo_ticker": request.yahoo_ticker,
            "company_name": request.company_name,
            "exchange": request.exchange,
            "download_status": "failed",
            "history_status": "unavailable",
            "batch_number": batch_number,
            "attempts": attempts,
            "n_observations": 0,
            "first_date": "",
            "last_date": "",
            "missing_vendor_fields": "|".join(sorted(REQUIRED_VENDOR_FIELDS)),
            "raw_path": "",
            "download_id": download_id,
            "attempted_at_utc": elapsed_start_timestamp(started),
            "completed_at_utc": utc_now(),
            "duration_seconds": round(time.monotonic() - started, 3),
            "error": error[:1000],
        }
    )


def _attempt_batch(
    batch: list[SecurityRequest],
    downloader: Downloader,
    config: dict[str, Any],
    start_date: str,
    end_date_exclusive: str,
    download_id: str,
    raw_dir: Path,
    journal: AcquisitionJournal,
    batch_number: int,
) -> None:
    if not batch:
        return
    retries = int(config["download"].get("retries", 3))
    backoff = float(config["download"].get("backoff_seconds", 5.0))
    pause = float(config["download"].get("request_pause_seconds", 0.5))
    timeout = int(config["download"].get("timeout_seconds", 45))
    pending = {request.yahoo_ticker: request for request in batch}
    attempts: dict[str, int] = {ticker: 0 for ticker in pending}
    last_error = ""
    batch_started = time.monotonic()

    for attempt in range(1, retries + 1):
        tickers = list(pending)
        attempts.update({ticker: attempt for ticker in tickers})
        batch_errors: dict[str, str] = {}
        LOGGER.info(
            "Batch %s attempt %s/%s: %s tickers",
            batch_number,
            attempt,
            retries,
            len(tickers),
        )
        try:
            data = downloader(
                tickers,
                start=start_date,
                end=end_date_exclusive,
                timeout=timeout,
            )
        except Exception as exc:  # A batch failure must not lose other tickers.
            last_error = f"{type(exc).__name__}: {exc}"
            LOGGER.warning("Batch %s failed: %s", batch_number, last_error)
            data = pd.DataFrame()
            batch_errors = vendor_errors(tickers)
        else:
            batch_errors = vendor_errors(tickers)

        for ticker in list(pending):
            request = pending[ticker]
            started = time.monotonic()
            frame = _extract_ticker_frame(data, ticker, len(tickers))
            normalised, missing = normalize_vendor_frame(
                frame,
                request,
                start_date,
                end_date_exclusive,
                download_id,
            )
            if normalised.empty:
                last_error = batch_errors.get(
                    ticker.upper(), "no observations returned"
                )
                continue
            raw_path = raw_output_path(raw_dir, request)
            atomic_write_parquet(normalised, raw_path)
            _record_success(
                request,
                normalised,
                missing,
                raw_path,
                journal,
                download_id,
                batch_number,
                attempt,
                started,
                start_date,
                end_date_exclusive,
            )
            pending.pop(ticker, None)
        journal.flush()
        if not pending:
            break
        permanently_unavailable = [
            ticker
            for ticker in pending
            if _is_permanent_request_unavailable(
                pending[ticker], batch_errors.get(ticker.upper(), "")
            )
        ]
        for ticker in permanently_unavailable:
            _record_failure(
                pending[ticker],
                journal,
                download_id,
                batch_number,
                attempt,
                batch_started,
                batch_errors[ticker.upper()],
            )
            pending.pop(ticker)
        if permanently_unavailable:
            LOGGER.info(
                "Batch %s: recorded %s permanent Yahoo missing/unavailable symbols",
                batch_number,
                len(permanently_unavailable),
            )
            journal.flush()
        if not pending:
            break
        if all(
            "too many requests" in batch_errors.get(ticker.upper(), "").lower()
            or "rate limited" in batch_errors.get(ticker.upper(), "").lower()
            for ticker in pending
        ):
            LOGGER.error(
                "Batch %s is rate limited; per-ticker retries would add request "
                "pressure and are skipped.",
                batch_number,
            )
            for ticker, request in pending.items():
                _record_failure(
                    request,
                    journal,
                    download_id,
                    batch_number,
                    attempt,
                    batch_started,
                    batch_errors[ticker.upper()],
                )
            pending.clear()
            journal.flush()
            return
        if attempt < retries:
            time.sleep(backoff * attempt)

    # Retry missing tickers individually so one bad symbol cannot hide or
    # invalidate the rest of a batch.
    for ticker, request in list(pending.items()):
        ticker_started = time.monotonic()
        last_error = "no observations returned"
        for attempt in range(1, retries + 1):
            attempts[ticker] = retries + attempt
            try:
                data = downloader(
                    [ticker],
                    start=start_date,
                    end=end_date_exclusive,
                    timeout=timeout,
                )
                frame = _extract_ticker_frame(data, ticker, 1)
                normalised, missing = normalize_vendor_frame(
                    frame,
                    request,
                    start_date,
                    end_date_exclusive,
                    download_id,
                )
                if not normalised.empty:
                    raw_path = raw_output_path(raw_dir, request)
                    atomic_write_parquet(normalised, raw_path)
                    _record_success(
                        request,
                        normalised,
                        missing,
                        raw_path,
                        journal,
                        download_id,
                        batch_number,
                        attempts[ticker],
                        ticker_started,
                        start_date,
                        end_date_exclusive,
                    )
                    pending.pop(ticker, None)
                    break
                last_error = vendor_errors([ticker]).get(
                    ticker.upper(), "no observations returned"
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                last_error = vendor_errors([ticker]).get(ticker.upper(), last_error)
                LOGGER.warning("Ticker %s attempt failed: %s", ticker, last_error)
            if attempt < retries:
                time.sleep(backoff * attempt)
        if ticker in pending:
            _record_failure(
                request,
                journal,
                download_id,
                batch_number,
                attempts[ticker],
                ticker_started,
                last_error,
            )
        journal.flush()
        time.sleep(pause)

    LOGGER.info(
        "Batch %s complete in %.1fs; %s failed",
        batch_number,
        time.monotonic() - batch_started,
        len(pending),
    )


def acquire(
    requests: list[SecurityRequest],
    config: dict[str, Any],
    *,
    raw_dir: Path | None = None,
    manifest_path: Path | None = None,
    failures_path: Path | None = None,
    resume: bool = True,
    downloader: Downloader = yahoo_download,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Acquire all requests in batches and return manifest/failure tables."""
    if not requests:
        raise ValueError("No securities selected for acquisition.")
    download_config = config["download"]
    batch_size = int(download_config.get("batch_size", 50))
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    raw_dir = raw_dir or project_path(config["paths"]["raw_dir"])
    metadata_dir = project_path(config["paths"]["metadata_dir"])
    manifest_path = manifest_path or metadata_dir / "download_manifest.csv"
    failures_path = failures_path or metadata_dir / "download_failures.csv"
    raw_dir.mkdir(parents=True, exist_ok=True)
    downloaded_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    download_id = f"{downloaded_at}-{uuid.uuid4().hex[:8]}"
    start_date = config["project"]["start_date"]
    end_date_exclusive = config["project"]["end_date_exclusive"]
    write_acquisition_provenance(config, metadata_dir, start_date, end_date_exclusive)
    journal = AcquisitionJournal(manifest_path, failures_path)
    invalid_dir = raw_dir.parent / "yahoo_invalid"

    pending: list[SecurityRequest] = []
    for request in requests:
        existing = journal.get(request.yahoo_ticker)
        if resume and existing and existing.get("download_status") in {
            "success",
            "partial",
        }:
            raw_path = (
                project_path(existing["raw_path"]) if existing.get("raw_path") else None
            )
            if raw_path and raw_path.exists() and recover_manifest_from_raw(
                request, raw_path, download_id
            ):
                LOGGER.info("Resume: keeping existing checkpoint for %s", request.yahoo_ticker)
                continue
            if raw_path and raw_path.exists():
                invalid_dir.mkdir(parents=True, exist_ok=True)
                quarantine = invalid_dir / (
                    f"{raw_path.stem}__{download_id}{raw_path.suffix}"
                )
                os.replace(raw_path, quarantine)
                LOGGER.warning(
                    "Resume: quarantined all-NaN checkpoint for %s at %s",
                    request.yahoo_ticker,
                    quarantine,
                )
        if resume:
            recovered_path = raw_output_path(raw_dir, request)
            recovered = recover_manifest_from_raw(
                request, recovered_path, download_id
            )
            if recovered:
                LOGGER.info(
                    "Resume: recovering raw checkpoint for %s", request.yahoo_ticker
                )
                journal.record(recovered)
                continue
        pending.append(request)

    journal.flush()
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        batch_number = offset // batch_size + 1
        _attempt_batch(
            batch,
            downloader,
            config,
            start_date,
            end_date_exclusive,
            download_id,
            raw_dir,
            journal,
            batch_number,
        )
        if offset + batch_size < len(pending):
            time.sleep(float(download_config.get("request_pause_seconds", 0.5)))
    journal.flush()
    return journal.manifest.copy(), pd.read_csv(failures_path, low_memory=False)


def parse_tickers(value: str | None) -> list[str]:
    """Parse comma/space separated ticker input."""
    if not value:
        return []
    return [part for part in re.split(r"[\s,]+", value) if part]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and preserve raw Yahoo Finance equity observations."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--universe",
        type=Path,
        default=Path("data/universe/security_master.parquet"),
        help="Security master path relative to the project root.",
    )
    parser.add_argument(
        "--tickers",
        help="Comma-separated smoke-test tickers, explicitly not a research universe.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--failures", type=Path, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload selected tickers instead of resuming checkpoints.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected security count and download parameters only.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise SystemExit("--batch-size must be positive")
        config["download"]["batch_size"] = args.batch_size
    log_path = project_path(config["paths"]["logs_dir"]) / "download_data.log"
    configure_logging(log_path)
    requests = load_requests(
        args.universe,
        limit=args.limit,
        explicit_tickers=parse_tickers(args.tickers),
    )
    LOGGER.info(
        "Selected %s securities, %s-%s exclusive",
        len(requests),
        config["project"]["start_date"],
        config["project"]["end_date_exclusive"],
    )
    if args.dry_run:
        print(
            f"securities={len(requests)} batch_size="
            f"{config['download']['batch_size']} raw_dir="
            f"{args.output_dir or config['paths']['raw_dir']}"
        )
        return 0
    manifest, failures = acquire(
        requests,
        config,
        raw_dir=args.output_dir,
        manifest_path=args.manifest,
        failures_path=args.failures,
        resume=not args.overwrite,
    )
    successful = int(
        manifest["download_status"].isin(["success", "partial"]).sum()
    )
    print(
        f"attempted={len(requests)} successful={successful} "
        f"failed_or_partial={len(failures)} "
        f"observations={int(pd.to_numeric(manifest['n_observations'], errors='coerce').fillna(0).sum())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
