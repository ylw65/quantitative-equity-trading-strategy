from pathlib import Path
import copy
import numpy as np,pandas as pd,pytest
from strategy import config,signals,target_schedule,load_market
from ledger import simulate

def toy(n=40):
    dates=pd.bdate_range('2020-01-01',periods=n);p=pd.DataFrame(np.column_stack([100*np.exp(-np.arange(n)*.02),100*np.exp(np.arange(n)*.01)]),index=dates,columns=['a','b']);v=pd.DataFrame(100.,index=dates,columns=['a','b']);v.iloc[30:]=300
    return {'dates':dates,'p':p,'v':v,'allowed':np.ones((n,2),bool),'tradable':np.ones((n,2),bool),'securities':['a','b'],'issuers':np.array([0,1])}
def event(t=0,direction=1,rv=2.,ret=-.1):return {'t':t,'j':0,'direction':direction,'rv':rv,'ret':ret,'strength':1.}
def test_no_invalid_revival_and_expiry():
    c=config();c['signal']['failure_exit']['event_low_break']=False;c['signal']['failure_exit']['no_rebound_sessions']=0;m=toy(10);f=pd.DataFrame([event()]);m['allowed'][1,0]=False
    w,_=target_schedule(m,f,c,0);assert w[0,0]>0 and not w[1:,0].any()
    m['allowed'][:]=True;c['signal']['max_age_sessions']=3
    w,_=target_schedule(m,f,c,0);assert w[:3,0].min()>0 and not w[3:,0].any()
def test_weak_signal_does_not_exit_and_opposite_does():
    c=config();c['signal']['failure_exit']['event_low_break']=False;c['signal']['failure_exit']['no_rebound_sessions']=0;m=toy(10);f=pd.DataFrame([event(),event(1,rv=1.1,ret=-.01),event(3,direction=-1,ret=.1)])
    w,_=target_schedule(m,f,c,0);assert w[:3,0].min()>0 and not w[3:,0].any()
def test_weak_opposite_does_not_exit():
    c=config();c['signal']['failure_exit']['event_low_break']=False;m=toy(10)
    f=pd.DataFrame([event(),event(2,direction=-1,rv=1.1,ret=.02)])
    w,_=target_schedule(m,f,c,0)
    assert w[:10,0].min()>0
def test_configuration_changes_budget_and_windows():
    c=config();m=toy();f=pd.DataFrame([event()]);a,_=target_schedule(m,f,c,0)
    cc=copy.deepcopy(c);cc['portfolio']['long_budget']=.5;b,_=target_schedule(m,f,cc,0)
    assert a[0,0]==2*b[0,0]
    cc['signal']['adv_days']=5;cc['signal']['reversal_days']=2;cc['signal']['history_days']=5
    assert not signals(m,c).equals(signals(m,cc))
def test_future_prices_do_not_change_earlier_signals():
    c=config();m=toy(65);a=signals(m,c);changed={**m,'p':m['p'].copy(),'v':m['v'].copy()};changed['p'].iloc[45:]*=10;changed['v'].iloc[45:]*=50
    b=signals(changed,c);pd.testing.assert_frame_equal(a[a.t<45],b[b.t<45])
def test_nonrebalance_preserves_units_and_next_close():
    c=config();m=toy(10);m['p'][:]=100;w=np.zeros((10,2));w[1:6,0]=.05;scheduled=np.zeros(10,bool)
    d,tr,h,orders=simulate(m,w,c,0,scheduled,lambda t,current:w[t],0,True)
    assert list(tr.date)==[m['dates'][2],m['dates'][7]]
    assert d.iloc[2:7].exposure.min()>0 and d.iloc[7].exposure==0
def test_dates_and_frozen_exit_reported():
    c=config();m=toy(10);w=np.zeros((10,2));w[1:,0]=.05;scheduled=np.zeros(10,bool)
    future=copy.deepcopy(m);future['dates']=pd.bdate_range('2024-01-01',periods=10)
    with pytest.raises(ValueError,match='test is forbidden'):simulate(future,w,c,0,scheduled,lambda t,current:w[t],0)
    m['tradable'][-1,0]=False
    d,_,_,_=simulate(m,w,c,0,scheduled,lambda t,current:w[t],0)
    assert d.untradable_value.iloc[-1]>0 and d.exposure.iloc[-1]>0
def test_real_snapshot_date_boundary_and_policy():
    c=config();m=load_market(c);assert m['dates'].max()<=pd.Timestamp('2022-12-31');assert c['allocation']=={'passive_core':.6,'active_satellite':.4,'transfers':'none'}
    assert c['policy']['final_selected'] and c['ml']['enabled']
    assert c['ml']['entry_quantile']==.5 and not c['test_enabled']

def test_data_current_identity_and_schema():
    import pyarrow.dataset as ds
    from market import dataset_directory
    from strategy import ROOT
    c=config();directory=dataset_directory(ROOT,c)
    assert {'open','high','low','close','adj_close','volume','dividends','stock_splits'}<=set(ds.dataset(directory/'selected_equity_daily.parquet').schema.names)
    m=load_market(c)
    assert len(m['securities'])==486
    assert 'SEC:0000048898' not in m['securities']

def test_source_rejects_future_and_external_path():
    from market import load_market,dataset_directory
    from strategy import ROOT
    c=config();c['allowed_end']='2023-01-03'
    with pytest.raises(ValueError):load_market(ROOT,c)
    c=config();c['data_directory']='../outside'
    with pytest.raises(ValueError):dataset_directory(ROOT,c)

def test_event_low_exit_and_cooldown_execute_next_close():
    c=config();c['signal']['failure_exit']['no_rebound_sessions']=0;c['signal']['failure_exit']['cooldown_sessions']=10
    m=toy(15);m['p'].iloc[:,0]=[100,101,98,97,96,95,94,93,92,91,90,89,88,87,86]
    m['low']=m['p'].copy();m['low'].iloc[0,0]=99
    f=pd.DataFrame([event(t) for t in (0,2,3,4,5,6,7,8,9,10)])
    w,scheduled=target_schedule(m,f,c,0)
    assert w[0,0]>.0 and w[2:,0].sum()==0
    d,tr,_,_=simulate(m,w,c,0,scheduled,lambda t,current:w[t],0,True)
    assert list(tr.loc[tr.trade_type.eq('entry'),'date'])==[m['dates'][1]]
    assert list(tr.loc[tr.trade_type.eq('exit'),'date'])==[m['dates'][3]]

def test_breadth_budget_scales_existing_positions():
    c=config();c['signal']['failure_exit']['event_low_break']=False;c['signal']['failure_exit']['no_rebound_sessions']=0
    c['signal']['market_breadth_budget']['enabled']=True
    c['signal']['market_breadth_budget']['lookback_sessions']=2
    m=toy(10);m['p'].iloc[:,0]=[100,100,90,85,84,83,82,81,80,79]
    m['p'].iloc[:,1]=[100,100,90,85,84,83,82,81,80,79]
    f=pd.DataFrame([event()]);w,scheduled=target_schedule(m,f,c,0)
    assert w[1,0]==.05 and w[2,0]==.0125 and scheduled[2]
