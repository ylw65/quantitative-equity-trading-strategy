"""Repository-owned reconstruction and evaluation of the selected final strategy."""
import copy
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import strategy as st
import features as ft
import ml
from market import read_daily, load_market, dataset_directory
from ledger import simulate

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str),
                    encoding='utf-8')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def subset(m, start, eligible=None):
    market = {
        key: (value.iloc[start:].copy() if isinstance(value, pd.DataFrame)
              else value[start:].copy() if isinstance(value, np.ndarray) and value.ndim == 2
              else value[start:] if key == 'dates' else value)
        for key, value in m.items()
    }
    if eligible is not None:
        market['allowed'] &= eligible
        market['tradable'] &= eligible
    return market


def schedules(m, events, c, start, eligible=None):
    market = subset(m, start, eligible)
    scoped = events[events.t >= start].copy()
    scoped.t -= start
    if eligible is not None:
        scoped = scoped[scoped.j.isin(np.flatnonzero(eligible))]
    return market, [st.target_schedule(market, scoped, c, phase)
                    for phase in c['selection']['phases']]


def panel(m, events, raw, shadow, start):
    union = np.logical_or.reduce([w > 0 for w, _ in shadow])
    ts, js = np.where(union)
    x = ft.features(m, events, raw, ts + start, js)
    latest = np.full(m['p'].shape, -1, int)
    for row in events[events.direction > 0].itertuples():
        latest[row.t, row.j] = row.t
    np.maximum.accumulate(latest, axis=0, out=latest)
    x['event_age'] = ts + start - latest[ts + start, js]  # Audit only, not a model feature.
    x['t'], x['j'], x['date'] = ts, js, m['dates'][ts + start]
    return x


def entry_plan(w, prob, threshold, adv, minimum):
    factors = np.full(w.shape[1], np.nan)

    def plan(t, current):
        factors[w[t] <= 0] = np.nan
        new = np.flatnonzero((w[t] > 0) & np.isnan(factors))
        if len(new):
            if not np.isfinite(prob[t, new]).all():
                raise ValueError('Missing causal probability at a proposed entry')
            factors[new] = ((prob[t, new] > threshold[t, new]) &
                            (adv[t, new] >= minimum)).astype(float)
        return w[t] * np.nan_to_num(factors)
    return plan


def passive(m, fee, initial):
    prices = m['p'].ffill().to_numpy()
    good = m['tradable'][0] & np.isfinite(prices[0])
    if not good.any():
        raise ValueError('No initially tradable passive securities')
    units = np.zeros(prices.shape[1])
    units[good] = initial / (1 + fee) / good.sum() / prices[0, good]
    return np.nan_to_num(prices * units).sum(axis=1)


def evaluate(m, shadow, c, adv, prob, threshold, fee):
    initial = c['portfolio']['initial_nav']
    active = np.zeros(len(m['dates']))
    equity = active.copy()
    cost, risk = 0., 0.
    phase_rows = []
    for phase, (w, scheduled) in zip(c['selection']['phases'], shadow):
        plan = entry_plan(w, prob, threshold, adv, c['ml']['prior_adv_minimum'])
        daily, _, _, _ = simulate(m, w, c, fee, scheduled.copy(), plan, 0)
        nav = daily.nav.to_numpy()
        active += nav / len(shadow)
        equity += nav * daily.exposure.to_numpy() / len(shadow)
        cost += daily.cost.sum() / len(shadow)
        risk = max(risk, float(daily.risk_breach.max()))
        phase_rows.append({'phase': phase, **st.stats(daily.daily_return)})
    core = passive(m, fee, initial)
    a = c['allocation']
    nav = a['passive_core'] * core + a['active_satellite'] * active
    stock = a['passive_core'] * core + a['active_satellite'] * equity
    ret = st.returns(nav, initial)
    summary = {
        **st.stats(ret), 'total_return': float(nav[-1] / initial - 1),
        'average_stock': float(np.mean(stock / nav)),
        'cost': float(a['passive_core'] * (initial - core[0]) + a['active_satellite'] * cost),
        'active_post_gap_cap_breach': risk,
    }
    daily = pd.DataFrame({'date': m['dates'], 'nav': nav, 'active_nav': active,
                          'core_nav': core, 'return': ret, 'cash': nav - stock,
                          'equity_exposure': stock / nav})
    return summary, daily, phase_rows


def build_teacher(m, events, c, raw, adv):
    """Reconstruct cashflow labels; naturally open trades are right-censored."""
    if m['dates'].max() > pd.Timestamp('2022-12-31'):
        raise ValueError('Never build learning labels from validation/test')
    market, shadow = schedules(m, events, c, 0)
    lookup = panel(m, events, raw, shadow, 0).set_index(['t', 'j'])
    records = []
    prices = m['p'].to_numpy()
    for phase, (w, scheduled) in enumerate(shadow):
        _, trades, _, _ = simulate(market, w, c, c['ml']['label_transaction_cost'],
                                    scheduled.copy(), lambda t, current: w[t], 0, True)
        for j, group in trades.groupby('j'):
            weights = w[:, j]
            starts = (weights > 0) & np.r_[True, weights[:-1] <= 0]
            segment = np.maximum.accumulate(np.where(starts, np.arange(len(weights)), -1))
            state = None
            for row in group.sort_values('date').itertuples():
                t = m['dates'].get_loc(row.date)
                if row.trade_type == 'entry':
                    intent = int(segment[m['dates'].get_loc(row.decision_date)])
                    state = {'t': intent, 'entry': t, 'cash': 0., 'buy': 0.}
                if state is None:
                    continue
                state['cash'] -= row.notional + row.cost
                if row.notional > 0:
                    state['buy'] += row.notional + row.cost
                if row.trade_type == 'exit':
                    intent = state['t']
                    if intent >= 0 and adv[intent, j] >= c['ml']['prior_adv_minimum']:
                        x = lookup.loc[(intent, j)].to_dict()
                        roi = state['cash'] / state['buy']
                        mae = max(0., 1 - np.nanmin(prices[state['entry']:t + 1, j]) /
                                  prices[state['entry'], j])
                        records.append({**x, 'j': int(j), 'phase': phase,
                                        'label_end_date': row.date, 'roi': roi,
                                        'loss': max(-roi, 0.), 'mae': mae})
                    state = None
    return pd.DataFrame(records)


def prepare_market(c):
    raw = read_daily(ROOT, c)
    m = load_market(ROOT, c, raw)
    events = st.signals(m, c)
    raw['dv'] = raw.close * raw.volume
    adv = ft.matrix(raw, 'dv', m).shift(1).rolling(20, min_periods=20).mean().to_numpy()
    return raw, m, events, adv


def probability_grid(market, x, models, quantile):
    prob = np.full(market['p'].shape, np.nan)
    threshold = prob.copy()
    for year in sorted(x.date.dt.year.unique()):
        model = models[int(year)]
        if model['latest_label'] >= pd.Timestamp(f'{year}-01-01'):
            raise ValueError('Model label end must precede prediction year')
        mask = x.date.dt.year.eq(year)
        ts, js = x.loc[mask, 't'].to_numpy(int), x.loc[mask, 'j'].to_numpy(int)
        prob[ts, js] = ml.probability(model, x[mask])
        threshold[ts, js] = model['quantiles'][str(quantile)]
    return prob, threshold


def train():
    c = st.config()
    out = ROOT / 'results/training'
    out.mkdir(parents=True, exist_ok=True)
    raw, m, events, adv = prepare_market(c)
    labels = build_teacher(m, events, c, raw, adv)
    labels.to_parquet(out / 'training_episodes.parquet', index=False)
    models, audit = ml.fit_sequence(labels, c['ml'])
    pd.DataFrame(audit).to_csv(out / 'model_date_audit.csv', index=False)
    joblib.dump(models, out / 'annual_models.joblib')
    start = np.flatnonzero(m['dates'] >= pd.Timestamp(c['ml']['training_prediction_start']))[0]
    market, shadow = schedules(m, events, c, start)
    x = panel(m, events, raw, shadow, start)
    prob, threshold = probability_grid(market, x, models, c['ml']['entry_quantile'])
    rows = []
    for fee in [c['costs']['transaction'], *c['costs']['sensitivity']]:
        summary, daily, phases = evaluate(market, shadow, c, adv[start:], prob, threshold, fee)
        rows.append({'fee': fee, **summary})
        daily.to_csv(out / f'daily_{fee:.3f}.csv', index=False)
        pd.DataFrame(phases).to_csv(out / f'phases_{fee:.3f}.csv', index=False)
        print('TRAIN', fee, json.dumps(summary), flush=True)
    pd.DataFrame(rows).to_csv(out / 'comparison.csv', index=False)
    frozen = load_frozen(c)
    # A full data rebuild must reproduce the selected model's probabilities/gate.
    original = ml.probability(frozen, labels)
    rebuilt = ml.probability(models[2023], labels)
    if not np.allclose(original, rebuilt, rtol=0, atol=1e-10):
        raise ValueError('Rebuilt2023 classifier differs from selected frozen model')
    if abs(models[2023]['quantiles']['0.5'] - frozen['quantiles']['0.5']) > 1e-10:
        raise ValueError('Rebuilt entry threshold differs from selected final version')
    dump(out / 'summary.json', {
        'version': c['version'], 'period': ['2019-01-01', '2022-12-31'],
        'training_market_end': m['dates'].max(), 'labels': len(labels),
        'latest_label': labels.label_end_date.max(), 'test_rows': 0,
        'frozen2023_reproduced': True, 'results': rows,
    })


def load_frozen(c):
    manifest = json.loads((ROOT / 'models/manifest.json').read_text(encoding='utf-8'))
    canonical = json.dumps(c, sort_keys=True, ensure_ascii=False).encode('utf-8')
    if hashlib.sha256(canonical).hexdigest() != manifest['strategy_config_sha256']:
        raise ValueError('Configuration differs from the selected final strategy')
    for name, expected in manifest['files'].items():
        if sha(ROOT / 'models' / name) != expected:
            raise ValueError(f'Frozen model/universe changed: {name}')
    model = joblib.load(ROOT / c['ml']['frozen_model'])
    if model['latest_label'] >= pd.Timestamp('2023-01-01'):
        raise ValueError('Frozen model contains validation/test labels')
    return model


def validation():
    c = st.config()
    model = load_frozen(c)
    c['allowed_end'], c['validation_enabled'] = '2023-12-31', True
    out = ROOT / 'results/validation'
    out.mkdir(parents=True, exist_ok=True)
    raw, m, events, adv = prepare_market(c)
    start = np.flatnonzero(m['dates'].year == 2023)[0]
    uni = pd.read_csv(ROOT / c['ml']['frozen_universe']).set_index('security_id')
    strict = uni.eligible_100m.reindex(m['securities']).to_numpy(bool)
    rows = []
    for label, eligible in [('primary474', strict), ('delivered486', np.ones(len(strict), bool))]:
        market, shadow = schedules(m, events, c, start, eligible)
        x = panel(m, events, raw, shadow, start)
        prob, threshold = probability_grid(market, x, {2023: model}, c['ml']['entry_quantile'])
        for fee in [c['costs']['transaction'], *c['costs']['sensitivity']]:
            summary, daily, phases = evaluate(market, shadow, c, adv[start:], prob, threshold, fee)
            rows.append({'cohort': label, 'fee': fee, **summary})
            daily.to_csv(out / f'{label}_daily_{fee:.3f}.csv', index=False)
            pd.DataFrame(phases).to_csv(out / f'{label}_phases_{fee:.3f}.csv', index=False)
            print('VALIDATION', label, fee, json.dumps(summary), flush=True)
    pd.DataFrame(rows).to_csv(out / 'comparison.csv', index=False)
    dump(out / 'summary.json', {
        'version': c['version'], 'market_end': m['dates'].max(), 'test_rows': 0,
        'validation_status': 'PREVIOUSLY_EXPOSED_DEVELOPMENT_NOT_INDEPENDENT_TEST',
        'fit_on2023': False, 'threshold': model['quantiles']['0.5'], 'results': rows,
    })


def test():
    """One frozen-model evaluation on the newly authorised 2024–2025 period."""
    c = st.config()
    model = load_frozen(c)
    protocol = json.loads((ROOT / 'docs/test_protocol.json').read_text())
    for name, expected in protocol['hashes'].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f'Pre-test frozen input changed: {name}')
    c['allowed_end'], c['test_enabled'] = '2025-12-31', True
    out = ROOT / 'results/test'
    out.mkdir(parents=True, exist_ok=True)
    raw, m, events, adv = prepare_market(c)
    start = np.flatnonzero(m['dates'].year == 2024)[0]
    uni = pd.read_csv(ROOT / c['ml']['frozen_universe']).set_index('security_id')
    strict = uni.eligible_100m.reindex(m['securities']).to_numpy(bool)
    rows, yearly = [], []
    for label, eligible in [('primary474', strict), ('delivered486', np.ones(len(strict), bool))]:
        market, shadow = schedules(m, events, c, start, eligible)
        x = panel(m, events, raw, shadow, start)
        prob, threshold = probability_grid(market, x, {2024: model, 2025: model}, c['ml']['entry_quantile'])
        for fee in [c['costs']['transaction'], *c['costs']['sensitivity']]:
            summary, daily, phases = evaluate(market, shadow, c, adv[start:], prob, threshold, fee)
            rows.append({'cohort': label, 'fee': fee, **summary})
            daily.to_csv(out / f'{label}_daily_{fee:.3f}.csv', index=False)
            pd.DataFrame(phases).to_csv(out / f'{label}_phases_{fee:.3f}.csv', index=False)
            for year, group in daily.groupby(daily.date.dt.year):
                ret = group['return'].to_numpy()
                yearly.append({'cohort': label, 'fee': fee, 'year': int(year),
                               'total_return': float(np.prod(1 + ret) - 1), **st.stats(ret)})
            print('TEST', label, fee, json.dumps(rows[-1]), flush=True)
    pd.DataFrame(rows).to_csv(out / 'comparison.csv', index=False)
    pd.DataFrame(yearly).to_csv(out / 'yearly.csv', index=False)
    dump(out / 'summary.json', {
        'version': c['version'], 'period': ['2024-01-01', '2025-12-31'],
        'market_end': m['dates'].max(), 'fit_on2023_or_test': False,
        'model_latest_label': model['latest_label'],
        'model_policy': protocol['model_policy'], 'threshold': model['quantiles']['0.5'],
        'test_market_rows': int((raw.date >= pd.Timestamp('2024-01-01')).sum()),
        'results': rows,
    })


def run(stage='train'):
    threadpool_limits(2)
    if stage == 'train':
        train()
    elif stage == 'validation':
        validation()
    elif stage == 'test':
        test()
    else:
        raise ValueError('Only train/validation/test are supported')
