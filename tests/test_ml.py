import copy
import json
import numpy as np
import pandas as pd
import pytest
import ml
import features
import pipeline
from strategy import config
from ledger import simulate
from market import load_market


def labels():
    rng=np.random.default_rng(71)
    z=pd.DataFrame(rng.normal(size=(120,len(ml.FEATURES))),columns=ml.FEATURES)
    z['roi']=rng.normal(0,.04,len(z))
    z['date']=pd.bdate_range('2018-01-01',periods=len(z))
    z['label_end_date']=z.date+pd.Timedelta(days=10)
    z.loc[119,'label_end_date']=pd.Timestamp('2021-01-01')
    return z


def test_future_labels_do_not_fit_classifier_scaler_or_threshold():
    z=labels();settings=config()['ml'];raw=np.linspace(.3,.6,len(z))
    a=ml.calibrate(ml.fit_classifier(z,'2019-01-01',settings),z,raw,'2019-01-01',settings)
    zz=z.copy();zz.loc[119,ml.FEATURES]=1e10;zz.loc[119,'roi']=100;rr=raw.copy();rr[119]=.9999
    b=ml.calibrate(ml.fit_classifier(zz,'2019-01-01',settings),zz,rr,'2019-01-01',settings)
    assert a['rows']==119
    assert np.array_equal(a['classifier'].coef_,b['classifier'].coef_)
    assert np.array_equal(a['scaler'].mean_,b['scaler'].mean_)
    assert a['quantiles']==b['quantiles']
    assert a['calibration_latest_label']<pd.Timestamp('2019-01-01')


def test_rejected_segment_does_not_retry_or_resize_after_new_probability():
    w=np.array([[.05],[.05],[0],[.05],[.05]])
    p=np.array([[.4],[.8],[.8],[.8],[.1]])
    planner=pipeline.entry_plan(w,p,np.ones_like(p)*.5,np.ones_like(p)*30e6,20e6)
    assert planner(0,np.zeros(1))[0]==0
    assert planner(1,np.zeros(1))[0]==0
    assert planner(2,np.zeros(1))[0]==0
    assert planner(3,np.zeros(1))[0]==.05
    assert planner(4,np.zeros(1))[0]==.05


def test_next_fill_price_cannot_change_prior_submitted_shares():
    c=config();dates=pd.bdate_range('2023-01-03',periods=3)
    p=pd.DataFrame([10.,10.,10.],index=dates,columns=['A'])
    m={'dates':dates,'p':p,'tradable':np.ones(p.shape,bool),'allowed':np.ones(p.shape,bool),'issuers':np.array([0]),'securities':['A']}
    w=np.ones((3,1))*.05;scheduled=np.zeros(3,bool)
    a=simulate(m,w,c,.002,scheduled,lambda t,current:w[t],0,True)
    changed=copy.deepcopy(m);changed['p'].iloc[1:]=20
    b=simulate(changed,w,c,.002,scheduled,lambda t,current:w[t],0,True)
    assert a[3].iloc[0].submitted_shares==b[3].iloc[0].submitted_shares
    assert a[1].iloc[0].date==dates[1]
    assert b[0].risk_breach.iloc[1]>0


def test_test_window_rejected_before_any_parquet_read(tmp_path):
    with pytest.raises(ValueError,match='test is forbidden'):
        load_market(tmp_path,{'allowed_start':'2016-01-01','allowed_end':'2024-01-01','data_directory':'missing'})


def test_future_ohlcv_cannot_change_past_features():
    dates=pd.bdate_range('2022-01-03',periods=100)
    p=pd.DataFrame(np.linspace(10,15,100)[:,None],index=dates,columns=['A'])
    m={'dates':dates,'p':p,'v':p*0+1000,'low':p*.98,'securities':['A']}
    raw=pd.DataFrame({'date':dates,'security_id':'A','high':p.iloc[:,0].to_numpy()*1.02,'low':p.iloc[:,0].to_numpy()*.98,'close':p.iloc[:,0].to_numpy()})
    events=pd.DataFrame({'t':[0],'j':[0],'direction':[1],'rv':[2.],'ret':[-.1],'strength':[.8]})
    a=features.features(m,events,raw,np.array([70]),np.array([0]))
    changed=copy.deepcopy(m)
    for col in ['p','v','low']:changed[col].iloc[71:]*=50
    raw.loc[raw.index>=71,['high','low','close']]*=50
    b=features.features(changed,events,raw,np.array([70]),np.array([0]))
    pd.testing.assert_frame_equal(a,b)


def test_frozen_selected_model_and_manifest():
    c=config();model=pipeline.load_frozen(c)
    assert model['used']==ml.FEATURES
    assert model['rows']==12990
    assert model['latest_label']<pd.Timestamp('2023-01-01')
    assert abs(model['quantiles']['0.5']-.43657775028548135)<1e-14
