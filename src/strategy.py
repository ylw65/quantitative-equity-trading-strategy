"""Final volume-reversal rules and configuration; no model or data side effects."""
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from market import load_market as read_market
ROOT=Path(__file__).resolve().parents[1]

def config():
    c=yaml.safe_load((ROOT/'configs/strategy.yaml').read_text(encoding='utf-8'))
    if c['test_enabled'] or c['validation_enabled'] or c['allowed_end']!='2022-12-31':
        raise ValueError('Default configuration must remain training-only; validation requires explicit CLI')
    if c['allowed_start']!='2016-01-01':raise ValueError('Frozen training history begins2016')
    if c['allocation']!={'passive_core':.6,'active_satellite':.4,'transfers':'none'}:
        raise ValueError('Final version uses initial60/40 without transfers')
    if c['execution']!={'lag_sessions':1,'price':'adj_close','hold_units_between_rebalances':True,'terminal_liquidation':False}:
        raise ValueError('Only previous-close shares, next-close fills, mark-to-market end supported')
    if c['ml']['family']!='logistic_noage' or not c['ml']['enabled'] or c['ml']['entry_quantile']!=.5:
        raise ValueError('Final strategy is no-age ML with prior probability median gate')
    if c['selection']['phases']!=[0,1,2,3,4]:raise ValueError('Five equal initial phases required')
    if c['portfolio']['short_budget']!=0 or c['costs']['cash_yield']!=0 or c['costs']['borrow']!=0:
        raise ValueError('Long-only, unlevered, zero cash interest research')
    return c

def load_market(c):return read_market(ROOT,c)

def signals(m,c):
    s=c['signal'];p,v=m['p'],m['v'];a=s['adv_days'];r=s['reversal_days'];history=max(s['history_days'],a,r)
    rv=v/v.shift(1).rolling(a,min_periods=a).mean();ret=p/p.shift(r)-1
    good=p.notna().rolling(history+1,min_periods=history+1).sum().eq(history+1)
    good &= v.shift(1).gt(0).rolling(a,min_periods=a).sum().eq(a)
    valid=good.to_numpy()&m['allowed']&np.isfinite(rv.to_numpy())&np.isfinite(ret.to_numpy())&ret.ne(0).to_numpy()&rv.gt(1).to_numpy()
    ts,js=np.where(valid);f=pd.DataFrame({'t':ts,'j':js,'direction':-np.sign(ret.to_numpy()[ts,js]),'rv':rv.to_numpy()[ts,js],'ret':ret.to_numpy()[ts,js]})
    group=f.groupby(['t','direction']);f['vrank']=group.rv.rank(pct=True);f['rrank']=group.ret.transform(lambda x:x.abs().rank(pct=True))
    f['strength']=s['volume_rank_weight']*f.vrank+s['reversal_rank_weight']*f.rrank
    # Weak events in either direction do not change the active state.
    weak_long=(f.direction>0)&((f.rv<s['relative_volume_min'])|(f.rv>s['relative_volume_max'])|(f.ret>-s['drop_min']))
    weak_opposite=(f.direction<0)&((f.rv<s['opposite_relative_volume_min'])|(f.ret<s['opposite_rise_min']))
    return f[~(weak_long|weak_opposite)].copy()

def target_schedule(m,f,c,phase):
    n,k=m['p'].shape;w=np.zeros((n,k));scheduled=np.zeros(n,bool);states={};held=set()
    s=c['signal'];sel=c['selection'];events={int(t):g for t,g in f.groupby('t')}
    prices=m['p'].to_numpy();lows=m.get('low',m['p']).to_numpy()
    failure=s['failure_exit'];breadth=s['market_breadth_budget']
    entry={};cooldown={};previous_budget=1.0
    for t in range(n):
        budget=1.0
        lookback=breadth['lookback_sessions']
        if breadth['enabled'] and t>=lookback:
            past=prices[t-lookback];now=prices[t]
            valid=np.isfinite(past)&np.isfinite(now)&(past>0)&m['allowed'][t]
            fraction=np.mean(now[valid]/past[valid]-1<=-breadth['stock_drop']) if valid.any() else 0
            if fraction>=breadth['severe_fraction']:budget=breadth['severe_budget']
            elif fraction>=breadth['medium_fraction']:budget=breadth['medium_budget']
        if budget!=previous_budget:scheduled[t]=True
        previous_budget=budget
        for j,st in list(states.items()):
            if not m['allowed'][t,j] or t-st[0]>=s['max_age_sessions']:del states[j]
        for row in events.get(t,pd.DataFrame()).itertuples():
            j=int(row.j)
            if row.direction>0 and s['relative_volume_min']<=row.rv<=s['relative_volume_max'] and row.ret<=-s['drop_min'] and t>=cooldown.get(j,-1):states[j]=(t,float(row.strength))
            elif row.direction<0 and row.rv>=s['opposite_relative_volume_min'] and row.ret>=s['opposite_rise_min']:states.pop(j,None)
        held.intersection_update(states)
        for j in list(held):
            signal_day,entry_low=entry[j]
            # The signal at signal_day executes at the next close. Inspect the first
            # complete holding day only; any exit executes at the following close.
            if t<signal_day+2:continue
            entry_price=prices[signal_day+1,j]
            broken=failure['event_low_break'] and np.isfinite(entry_low) and prices[t,j]<entry_low
            no_rebound=failure['no_rebound_sessions']>0 and t>=signal_day+1+failure['no_rebound_sessions'] and prices[t,j]<=entry_price
            if broken or no_rebound:
                held.remove(j);states.pop(j,None);entry.pop(j,None)
                cooldown[j]=t+failure['cooldown_sessions']+1
        if t%sel['rebalance_sessions']==phase:
            scheduled[t]=True
            ranking=sorted(states,key=lambda j:(-states[j][1],m['securities'][j]));ranks={j:i+1 for i,j in enumerate(ranking)}
            held={j for j in held if ranks.get(j,np.inf)<=sel['retain_rank']}
            entry={j:entry[j] for j in held}
            for j in ranking[:sel['enter_rank']]:
                if len(held)>=sel['enter_rank']:break
                if j not in held:
                    held.add(j);entry[j]=(t,lows[states[j][0],j])
        for j in held:w[t,j]=budget*c['portfolio']['long_budget']/sel['enter_rank']
    return w,scheduled

def stats(r):
    r=np.asarray(r);v=r.std(ddof=1)*np.sqrt(252);wealth=np.cumprod(1+r);peak=np.maximum.accumulate(np.r_[1.,wealth])[1:]
    return {'annual_return':float(wealth[-1]**(252/len(r))-1),'volatility':float(v),'sharpe':float(r.mean()*252/v) if v else None,'max_drawdown':float(np.min(wealth/peak-1))}
def returns(nav,initial):return np.r_[nav[0]/initial-1,nav[1:]/nav[:-1]-1]
