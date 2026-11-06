"""Past-only profit classifier, sigmoid calibration and historical median gate."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

FEATURES = [
    'ret1', 'ret5', 'ret20', 'rv', 'clv', 'range', 'vol20', 'vol_ratio',
    'low_distance', 'relative5', 'market5', 'market20', 'breadth',
    'event_rv', 'event_ret', 'event_score', 'event_low_distance',
]


def date_weights(frame):
    weights = 1 / frame.groupby('date').date.transform('size').to_numpy()
    return weights / weights.mean()


def transform(model, frame):
    x = frame[model['used']].clip(model['lo'], model['hi'], axis=1)
    return model['scaler'].transform(model['imputer'].transform(x))


def fit_classifier(labels, cutoff, settings):
    history = labels[labels.label_end_date < pd.Timestamp(cutoff)].copy()
    if history.empty or history.roi.gt(0).nunique() != 2:
        raise ValueError('Insufficient completed pre-cutoff labels')
    x = history[FEATURES]
    lo, hi = [x.quantile(q) for q in settings['clip_quantiles']]
    imputer = SimpleImputer(keep_empty_features=True)
    xx = imputer.fit_transform(x.clip(lo, hi, axis=1))
    scaler = StandardScaler().fit(xx)
    classifier = LogisticRegression(
        C=settings['classifier_C'], max_iter=settings['max_iter'])
    classifier.fit(scaler.transform(xx), history.roi.gt(0),
                   sample_weight=date_weights(history))
    return {
        'used': FEATURES, 'lo': lo, 'hi': hi, 'imputer': imputer,
        'scaler': scaler, 'classifier': classifier, 'cutoff': cutoff,
        'rows': len(history), 'latest_label': history.label_end_date.max(),
        'calibrator': None, 'quantiles': {},
    }


def raw_probability(model, frame):
    return model['classifier'].predict_proba(transform(model, frame))[:, 1]


def calibrate(model, labels, prior_oof, cutoff, settings):
    mask = (labels.label_end_date < pd.Timestamp(cutoff)) & np.isfinite(prior_oof)
    history = labels.loc[mask]
    if history.empty:
        raise ValueError('No resolved prior-year out-of-fit calibration labels')
    p = np.clip(prior_oof[mask], 1e-6, 1 - 1e-6)
    logits = np.log(p / (1 - p))[:, None]
    calibrator = LogisticRegression(
        C=settings['calibration_C'], max_iter=settings['max_iter'])
    calibrator.fit(logits, history.roi.gt(0), sample_weight=date_weights(history))
    calibrated = calibrator.predict_proba(logits)[:, 1]
    q = settings['entry_quantile']
    model.update(
        calibrator=calibrator,
        quantiles={str(q): float(np.quantile(calibrated, q))},
        calibration_rows=len(history),
        calibration_latest_label=history.label_end_date.max(),
    )
    return model


def probability(model, frame):
    p = np.clip(raw_probability(model, frame), 1e-6, 1 - 1e-6)
    if model['calibrator'] is None:
        raise ValueError('The final strategy requires past-only calibration')
    return model['calibrator'].predict_proba(np.log(p / (1 - p))[:, None])[:, 1]


def fit_sequence(labels, settings):
    """Each annual prediction/calibration excludes not-yet-completed trades."""
    oof = np.full(len(labels), np.nan)
    models, audit = {}, []
    for year in range(2017, 2023):
        cutoff = f'{year}-01-01'
        model = fit_classifier(labels, cutoff, settings)
        current = labels.date.dt.year.eq(year).to_numpy()
        oof[current] = raw_probability(model, labels[current])
        if year >= 2019:
            model = calibrate(model, labels, oof, cutoff, settings)
            models[year] = model
            audit.append(model_audit(model, year))
    model = calibrate(fit_classifier(labels, '2023-01-01', settings),
                      labels, oof, '2023-01-01', settings)
    models[2023] = model
    audit.append(model_audit(model, 2023))
    return models, audit


def model_audit(model, year):
    return {
        'prediction_year': year, 'rows': model['rows'],
        'latest_label': model['latest_label'],
        'calibration_rows': model['calibration_rows'],
        'calibration_latest_label': model['calibration_latest_label'],
        'entry_threshold': model['quantiles']['0.5'],
    }
