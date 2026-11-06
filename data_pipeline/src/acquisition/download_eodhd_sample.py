"""Acquire a clearly labelled small EODHD public-demo audit sample.

This module is a secondary fallback for exercising market-data quality checks.
It is not a replacement for the full-universe Yahoo Finance acquisition.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import pandas as pd
import requests
import yaml

from src.common import (
    PROJECT_ROOT,
    atomic_write_json,
    configure_logging,
    ensure_parent,
    load_config,
    project_path,
)


LOGGER = logging.getLogger("us_equity_data.eodhd_sample")
PROVIDER = "eodhd_public_demo"

MANIFEST_COLUMNS = [
    "security_id",
    "ticker_2023",
    "yahoo_ticker",
    "eodhd_code",
    "company_name",
    "exchange",
    "security_type",
    "current_status",
    "selection_stratum",
    "download_status",
    "price_status",
    "split_status",
    "dividend_status",
    "n_observations",
    "first_date",
    "last_date",
    "n_splits",
    "n_dividends",
    "raw_price_path",
    "raw_split_path",
    "raw_dividend_path",
    "normalized_path",
    "attempted_at_utc",
    "completed_at_utc",
    "error",
]


@dataclass(frozen=True)
class SampleRequest:
    security_id: str
    ticker_2023: str
    yahoo_ticker: str
    eodhd_code: str
    company_name: str
    exchange: str
    security_type: str
    current_status: str
    selection_stratum: str


@dataclass(frozen=True)
class ApiResponse:
    status_code: int | None
    content: bytes
    from_cache: bool
    attempted_at_utc: str
    error: str = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _string(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _safe_ticker(ticker: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", ticker)


def _ticker_key(ticker: str) -> str:
    return hashlib.sha256(ticker.encode("utf-8")).hexdigest()[:10]


def _path_value(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def atomic_write_bytes(content: bytes, path: Path) -> None:
    target = ensure_parent(path)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    target = ensure_parent(path)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", encoding="utf-8", dir=target.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    try:
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
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


def _request_from_row(row: pd.Series) -> SampleRequest:
    ticker = _string(row.get("yahoo_ticker")).upper()
    current_status = (
        "current_sec_listing_match"
        if _string(row.get("current_ticker"))
        else "no_current_ticker_match"
    )
    stratum = "|".join(
        [
            _string(row.get("exchange")) or "Unknown",
            _string(row.get("security_type")) or "Unknown",
            current_status,
        ]
    )
    return SampleRequest(
        security_id=_string(row.get("security_id")),
        ticker_2023=_string(row.get("ticker_2023")).upper() or ticker,
        yahoo_ticker=ticker,
        eodhd_code=f"{ticker}.US",
        company_name=_string(row.get("company_name")),
        exchange=_string(row.get("exchange")),
        security_type=_string(row.get("security_type")),
        current_status=current_status,
        selection_stratum=stratum,
    )


def select_sample_requests(
    universe_path: str | Path,
    *,
    limit: int,
    seed: int,
    explicit_tickers: Iterable[str] = (),
) -> list[SampleRequest]:
    """Select a deterministic cross-stratum sample from the verified master."""
    path = project_path(universe_path)
    master = pd.read_parquet(path)
    required = {
        "security_id",
        "ticker_2023",
        "yahoo_ticker",
        "company_name",
        "exchange",
        "security_type",
    }
    missing = sorted(required - set(master.columns))
    if missing:
        raise ValueError(f"Security master lacks required columns: {missing}")
    master = master.copy()
    master["yahoo_ticker"] = master["yahoo_ticker"].astype(str).str.upper()
    if master["yahoo_ticker"].duplicated().any():
        raise ValueError("Security master contains duplicate Yahoo tickers")

    by_ticker: dict[str, SampleRequest] = {}
    for _, row in master.iterrows():
        request = _request_from_row(row)
        by_ticker[request.yahoo_ticker] = request
    explicit = list(dict.fromkeys(str(t).upper() for t in explicit_tickers if t))
    ordered: list[SampleRequest] = [
        by_ticker[ticker] for ticker in explicit if ticker in by_ticker
    ][:limit]
    selected = {request.yahoo_ticker for request in ordered}

    candidates = master[~master["yahoo_ticker"].isin(selected)].copy()
    candidates["_stratum"] = [
        _request_from_row(row).selection_stratum
        for _, row in candidates.iterrows()
    ]
    rng = __import__("numpy").random.default_rng(seed)
    pools: dict[str, list[pd.Series]] = {}
    for stratum, group in candidates.groupby("_stratum", sort=True):
        indexes = rng.permutation(len(group))
        pools[str(stratum)] = [group.iloc[i] for i in indexes]

    represented = {request.selection_stratum for request in ordered}
    while len(ordered) < limit:
        progressed = False
        # Complete first coverage of every stratum before taking a second
        # security from any stratum represented by the explicit picks.
        stratum_order = sorted(
            pools, key=lambda value: (value in represented, value)
        )
        for stratum in stratum_order:
            if len(ordered) >= limit:
                break
            pool = pools[stratum]
            if not pool:
                continue
            request = _request_from_row(pool.pop())
            if request.yahoo_ticker not in selected:
                selected.add(request.yahoo_ticker)
                ordered.append(request)
                represented.add(stratum)
                progressed = True
        if not progressed:
            break

    if not ordered:
        raise ValueError("No sample securities could be selected")
    return ordered


class EodhdClient:
    """Rate-limited EODHD client that caches every raw response body."""

    def __init__(
        self,
        api_base: str,
        token: str,
        requests_per_second: float,
        timeout_seconds: float,
        retries: int,
        backoff_seconds: float,
        raw_root: Path,
        overwrite: bool = False,
        session: requests.Session | None = None,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.token = token
        self.minimum_interval = 1.0 / max(requests_per_second, 0.1)
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.backoff_seconds = backoff_seconds
        self.raw_root = raw_root
        self.overwrite = overwrite
        self.session = session or requests.Session()
        self.session.headers.update(
            {"User-Agent": "DSA5205-Proj2 academic data-quality sample"}
        )
        self._last_request = 0.0

    def _pace(self) -> None:
        remaining = self.minimum_interval - (time.monotonic() - self._last_request)
        if remaining > 0:
            time.sleep(remaining)
        self._last_request = time.monotonic()

    def fetch(
        self,
        *,
        endpoint: str,
        code: str,
        suffix: str,
        params: dict[str, str],
    ) -> ApiResponse:
        raw_name = f"{_safe_ticker(code)}__{_ticker_key(code)}__{endpoint}{suffix}"
        raw_path = self.raw_root / "raw_responses" / raw_name
        attempted = utc_now()
        if raw_path.exists() and not self.overwrite:
            return ApiResponse(None, raw_path.read_bytes(), True, attempted)

        url = f"{self.api_base}/{endpoint}/{quote(code, safe='.-_')}"
        query = dict(params)
        query["api_token"] = self.token
        last_error = ""
        for attempt in range(1, self.retries + 1):
            self._pace()
            try:
                response = self.session.get(
                    url, params=query, timeout=self.timeout_seconds
                )
                content = response.content
                atomic_write_bytes(content, raw_path)
                if response.status_code == 200:
                    return ApiResponse(
                        response.status_code, content, False, attempted
                    )
                last_error = (
                    f"HTTP {response.status_code}: "
                    f"{content.decode('utf-8', errors='replace')[:300]}"
                )
                if response.status_code not in {429, 500, 502, 503, 504}:
                    break
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < self.retries:
                time.sleep(self.backoff_seconds * (2 ** (attempt - 1)))
        return ApiResponse(None, b"", False, attempted, last_error[:1000])


def _parse_json_events(content: bytes) -> list[dict[str, Any]]:
    if not content:
        raise ValueError("empty event response")
    payload = json.loads(content.decode("utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected a JSON event list, got {type(payload).__name__}")
    return [row for row in payload if isinstance(row, dict)]


def parse_price_csv(content: bytes) -> pd.DataFrame:
    """Parse and validate the EODHD OHLCV body without filling observations."""
    text = content.decode("utf-8", errors="replace").lstrip("\ufeff")
    if not text.lower().startswith("date,"):
        raise ValueError(f"unexpected price response: {text[:200]!r}")
    from io import StringIO

    frame = pd.read_csv(StringIO(text))
    renames = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adjusted_close": "adj_close",
        "Adj_close": "adj_close",
        "Adjusted_Close": "adj_close",
        "Volume": "volume",
    }
    frame = frame.rename(columns=renames)
    required = {"date", "open", "high", "low", "close", "adj_close", "volume"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"price response lacks columns: {missing}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in required - {"date"}:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[frame["date"].notna()].sort_values("date", kind="stable")
    if frame.empty:
        raise ValueError("price response contains no dated observations")
    # Any extra fields remain available in the preserved raw CSV. The normalized
    # table is restricted to the common validation contract.
    return frame[["date", "open", "high", "low", "close", "adj_close", "volume"]].copy()


def _split_factor(value: Any) -> float:
    text = str(value).strip()
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return float(numerator) / float(denominator)
    return float(text)


def _align_events(
    daily_dates: pd.Series,
    events: list[dict[str, Any]],
    value_column: str,
    converter: Any,
) -> tuple[pd.Series, int]:
    result = pd.Series(0.0, index=daily_dates.index, dtype="float64")
    accepted = 0
    session_dates = pd.DatetimeIndex(daily_dates)
    for event in events:
        event_date = pd.to_datetime(event.get("date"), errors="coerce")
        if pd.isna(event_date):
            continue
        try:
            value = converter(event.get(value_column))
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if not pd.notna(value) or value < 0:
            continue
        position = int(session_dates.searchsorted(event_date, side="left"))
        if position >= len(session_dates):
            continue
        result.iloc[position] += float(value)
        accepted += 1
    return result, accepted


def normalize_responses(
    request: SampleRequest,
    price_content: bytes,
    split_content: bytes,
    dividend_content: bytes,
    *,
    download_id: str,
) -> tuple[pd.DataFrame, int, int]:
    """Convert raw EODHD responses into the shared validation schema."""
    daily = parse_price_csv(price_content)
    split_events = _parse_json_events(split_content)
    dividend_events = _parse_json_events(dividend_content)
    splits, n_splits = _align_events(
        daily["date"], split_events, "split", _split_factor
    )
    dividends, n_dividends = _align_events(
        daily["date"],
        dividend_events,
        "unadjustedValue",
        float,
    )
    daily["dividends"] = dividends
    daily["stock_splits"] = splits
    daily.insert(1, "ticker", request.yahoo_ticker)
    daily.insert(2, "security_id", request.security_id)
    daily.insert(3, "ticker_2023", request.ticker_2023)
    daily.insert(4, "source", PROVIDER)
    daily.insert(5, "provider", PROVIDER)
    daily.insert(6, "download_id", download_id)
    daily.insert(7, "retrieved_at_utc", utc_now())
    return daily, n_splits, n_dividends


def _valid_event_content(content: bytes) -> bool:
    try:
        _parse_json_events(content)
        return True
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
        return False


def _write_provenance(
    config: dict[str, Any],
    metadata_dir: Path,
    requests_selected: int,
    manifest: pd.DataFrame,
) -> None:
    token = os.environ.get(config["source"]["token_env"])
    counts = manifest["download_status"].value_counts(dropna=False).to_dict()
    payload = {
        "provider": PROVIDER,
        "generated_at_utc": utc_now(),
        "empirical_role": config["scope"]["empirical_role"],
        "full_universe_replacement": config["scope"]["full_universe_replacement"],
        "sample_requested": config["sample"]["limit"],
        "sample_selected": requests_selected,
        "selection_seed": config["sample"]["seed"],
        "selection_method": (
            "Explicit representative tickers followed by deterministic "
            "round-robin sampling by exchange, security type, and current "
            "SEC-ticker status. No return, beta, or volume ranking was used."
        ),
        "api_base": config["source"]["api_base"],
        "token_source": (
            config["source"]["token_env"]
            if token
            else "public_demo_default"
        ),
        "period": config["period"],
        "request_parameters": {
            "requests_per_second": config["source"]["requests_per_second"],
            "retries": config["source"]["retries"],
            "timeout_seconds": config["source"]["timeout_seconds"],
        },
        "download_status_counts": counts,
        "limitations": config["scope"]["limitations"],
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "requests": requests.__version__,
            "yfinance": _package_version("yfinance"),
            "pyarrow": _package_version("pyarrow"),
        },
    }
    atomic_write_json(
        payload, metadata_dir / "eodhd_demo_provenance.json"
    )


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def run_download(
    config_path: str | Path,
    *,
    limit: int | None = None,
    overwrite: bool = False,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    if limit is not None:
        config["sample"]["limit"] = int(limit)
    universe_path = project_path(config["paths"]["universe"])
    raw_root = project_path(config["paths"]["raw_root"])
    metadata_dir = project_path(config["paths"]["metadata_dir"])
    raw_root.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(
        project_path(config["paths"].get("logs_dir", "logs"))
        / "eodhd_demo.log"
    )

    requests_to_make = select_sample_requests(
        universe_path,
        limit=int(config["sample"]["limit"]),
        seed=int(config["sample"]["seed"]),
        explicit_tickers=config["sample"].get("explicit_tickers", []),
    )
    token = os.environ.get(
        config["source"]["token_env"],
        config["source"]["public_demo_default"],
    )
    client = EodhdClient(
        config["source"]["api_base"],
        token,
        float(config["source"]["requests_per_second"]),
        float(config["source"]["timeout_seconds"]),
        int(config["source"]["retries"]),
        float(config["source"]["backoff_seconds"]),
        raw_root,
        overwrite=overwrite,
        session=session,
    )
    download_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + hashlib.sha256(str(time.time_ns()).encode()).hexdigest()[:8]
    )
    start = str(config["period"]["start_date"])
    end = str(config["period"]["end_date_exclusive"])
    common = {"fmt": "json", "from": start, "to": end}
    rows: list[dict[str, Any]] = []
    for number, request in enumerate(requests_to_make, start=1):
        LOGGER.info(
            "EODHD sample %d/%d: %s (%s)",
            number,
            len(requests_to_make),
            request.yahoo_ticker,
            request.selection_stratum,
        )
        price = client.fetch(
            endpoint="eod",
            code=request.eodhd_code,
            suffix=".csv",
            params={
                "fmt": "csv",
                "from": start,
                "to": end,
                "period": "d",
            },
        )
        price_error = price.error
        if not price_error and price.content:
            try:
                parse_price_csv(price.content)
                price_status = "success"
            except (ValueError, UnicodeDecodeError) as exc:
                price_status = "failed"
                price_error = str(exc)
        else:
            price_status = "failed"
            price_error = price_error or "empty price response"

        if price_status != "success":
            rows.append(
                {
                    "security_id": request.security_id,
                    "ticker_2023": request.ticker_2023,
                    "yahoo_ticker": request.yahoo_ticker,
                    "eodhd_code": request.eodhd_code,
                    "company_name": request.company_name,
                    "exchange": request.exchange,
                    "security_type": request.security_type,
                    "current_status": request.current_status,
                    "selection_stratum": request.selection_stratum,
                    "download_status": "failed",
                    "price_status": "failed",
                    "split_status": "not_attempted",
                    "dividend_status": "not_attempted",
                    "n_observations": 0,
                    "first_date": "",
                    "last_date": "",
                    "n_splits": 0,
                    "n_dividends": 0,
                    "raw_price_path": _path_value(
                        raw_root
                        / "raw_responses"
                        / (
                            f"{_safe_ticker(request.eodhd_code)}__"
                            f"{_ticker_key(request.eodhd_code)}__eod.csv"
                        )
                    ),
                    "raw_split_path": "",
                    "raw_dividend_path": "",
                    "normalized_path": "",
                    "attempted_at_utc": price.attempted_at_utc,
                    "completed_at_utc": utc_now(),
                    "error": price_error[:1000],
                }
            )
            continue

        split = client.fetch(
            endpoint="splits",
            code=request.eodhd_code,
            suffix=".json",
            params=common,
        )
        dividend = client.fetch(
            endpoint="div",
            code=request.eodhd_code,
            suffix=".json",
            params=common,
        )
        split_status = (
            "success"
            if not split.error and _valid_event_content(split.content)
            else "failed"
        )
        dividend_status = (
            "success"
            if not dividend.error and _valid_event_content(dividend.content)
            else "failed"
        )
        normalized_path = (
            raw_root
            / "daily"
            / (
                f"{_safe_ticker(request.yahoo_ticker)}__"
                f"{_ticker_key(request.security_id + '|' + request.yahoo_ticker)}"
                ".parquet"
            )
        )
        error_parts = []
        if split_status != "success":
            error_parts.append(f"splits: {split.error or 'invalid JSON'}")
        if dividend_status != "success":
            error_parts.append(f"dividends: {dividend.error or 'invalid JSON'}")
        n_splits = n_dividends = 0
        if not error_parts:
            try:
                normalized, n_splits, n_dividends = normalize_responses(
                    request,
                    price.content,
                    split.content,
                    dividend.content,
                    download_id=download_id,
                )
                atomic_write_parquet(normalized, normalized_path)
                status = "success"
                n_obs = len(normalized)
                first = normalized["date"].min().date().isoformat()
                last = normalized["date"].max().date().isoformat()
                normalized_value = _path_value(normalized_path)
            except Exception as exc:  # preserve per-security failure
                status = "failed"
                n_obs = 0
                first = last = ""
                normalized_value = ""
                error_parts.append(f"normalization: {type(exc).__name__}: {exc}")
        else:
            status = "partial"
            normalized, _, _ = normalize_responses(
                request,
                price.content,
                b"[]" if split_status != "success" else split.content,
                b"[]" if dividend_status != "success" else dividend.content,
                download_id=download_id,
            )
            atomic_write_parquet(normalized, normalized_path)
            n_obs = len(normalized)
            first = normalized["date"].min().date().isoformat()
            last = normalized["date"].max().date().isoformat()
            normalized_value = _path_value(normalized_path)

        rows.append(
            {
                "security_id": request.security_id,
                "ticker_2023": request.ticker_2023,
                "yahoo_ticker": request.yahoo_ticker,
                "eodhd_code": request.eodhd_code,
                "company_name": request.company_name,
                "exchange": request.exchange,
                "security_type": request.security_type,
                "current_status": request.current_status,
                "selection_stratum": request.selection_stratum,
                "download_status": status,
                "price_status": price_status,
                "split_status": split_status,
                "dividend_status": dividend_status,
                "n_observations": n_obs,
                "first_date": first,
                "last_date": last,
                "n_splits": n_splits,
                "n_dividends": n_dividends,
                "raw_price_path": _path_value(
                    raw_root
                    / "raw_responses"
                    / (
                        f"{_safe_ticker(request.eodhd_code)}__"
                        f"{_ticker_key(request.eodhd_code)}__eod.csv"
                    )
                ),
                "raw_split_path": _path_value(
                    raw_root
                    / "raw_responses"
                    / (
                        f"{_safe_ticker(request.eodhd_code)}__"
                        f"{_ticker_key(request.eodhd_code)}__splits.json"
                    )
                ),
                "raw_dividend_path": _path_value(
                    raw_root
                    / "raw_responses"
                    / (
                        f"{_safe_ticker(request.eodhd_code)}__"
                        f"{_ticker_key(request.eodhd_code)}__div.json"
                    )
                ),
                "normalized_path": normalized_value,
                "attempted_at_utc": price.attempted_at_utc,
                "completed_at_utc": utc_now(),
                "error": "; ".join(error_parts)[:1000],
            }
        )

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    manifest_path = metadata_dir / "eodhd_demo_manifest.csv"
    failures_path = metadata_dir / "eodhd_demo_failures.csv"
    atomic_write_csv(manifest, manifest_path)
    atomic_write_csv(
        manifest[manifest["download_status"].isin(["failed", "partial"])],
        failures_path,
    )
    _write_provenance(config, metadata_dir, len(requests_to_make), manifest)
    result = {
        "selected": len(requests_to_make),
        "success": int(manifest["download_status"].eq("success").sum()),
        "partial": int(manifest["download_status"].eq("partial").sum()),
        "failed": int(manifest["download_status"].eq("failed").sum()),
        "observations": int(manifest["n_observations"].sum()),
        "manifest": str(manifest_path),
        "failures": str(failures_path),
    }
    LOGGER.info("EODHD sample result: %s", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eodhd_demo.yaml")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    result = run_download(
        config_path,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
