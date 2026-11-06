"""Previous-close share orders, next-close fills; no execution-price target sizing."""
import numpy as np,pandas as pd

def simulate(m,w,c,fee,scheduled,plan,start,capture=False):
    if m['dates'].max()>pd.Timestamp('2025-12-31'):
        raise ValueError('Dates after the authorised test period are forbidden')
    if m['dates'].max()>pd.Timestamp('2023-12-31') and not c.get('test_enabled',False):
        raise ValueError('2024–2025 test is forbidden without explicit test mode')
    dates=m['dates'];prices=m['p'].to_numpy();k=prices.shape[1];cash=previous=float(c['portfolio']['initial_nav']);q=np.zeros(k);mark=np.zeros(k);pending=np.zeros(k);decision=None;rows=[];trades=[];holds=[];orders=[]
    from engine import project
    for t in range(start,len(dates)):
        date=dates[t];today=prices[t];known=np.isfinite(today)&(today>0);mark[known]=today[known];trad=m['tradable'][t];before_values=q*mark;pre=cash+before_values.sum();delta=np.zeros(k)
        # Submitted shares are immutable across the overnight interval. Cash-only
        # partial fills cap buys at realized funds; no new alpha decision at fill.
        sell=np.where(trad,np.minimum(0,np.maximum(pending,-q)),0);sale=-sell*mark;cash+=sale.sum()*(1-fee);q+=sell;delta+=sell
        buy=np.where(trad,np.maximum(pending,0),0);request=float((buy*mark).sum()*(1+fee));scale=min(1,max(0,cash)/request) if request else 1.;buy*=scale;cash-=(buy*mark).sum()*(1+fee);q+=buy;delta+=buy
        cost=float((np.abs(delta)*mark).sum()*fee);values=q*mark;nav=float(cash+values.sum());assert cash>=-1e-5 and np.isclose(nav,pre-cost,rtol=1e-10,atol=1e-5)
        actual=values/nav;company=np.bincount(m['issuers'],weights=np.abs(actual));breach=max(actual.max()-.05,company.max()-.05,actual.sum()-1,0)
        rows.append({'date':date,'nav':nav,'daily_return':nav/previous-1,'cash':float(cash),'exposure':float(actual.sum()),'cost':cost,'trade_notional':float((np.abs(delta)*mark).sum()),'cash_fill_scale':float(scale),'risk_breach':float(breach),'untradable_value':float(values[~trad].sum())})
        if capture:
            for j in np.flatnonzero(np.abs(delta)*mark>nav*1e-10):
                kind='entry' if abs(before_values[j])<nav*1e-10 else ('exit' if abs(values[j])<nav*1e-10 else 'resize')
                trades.append({'date':date,'decision_date':decision,'security_id':m['securities'][j],'j':int(j),'notional':float(delta[j]*mark[j]),'cost':float(abs(delta[j])*mark[j]*fee),'trade_type':kind})
            for j in np.flatnonzero(actual>1e-10):holds.append({'date':date,'j':int(j),'weight':float(actual[j]),'value':float(values[j]),'units':float(q[j])})
        previous=nav
        if t==len(dates)-1:continue  # no order beyond the evaluation interval
        target=np.asarray(plan(t,actual.copy()),float);assert np.isfinite(target).all() and (target>=0).all()
        frozen=np.where(trad,0,values);want=np.where(trad,target*nav,0);continuing=(target*values>0)&trad&(not bool(scheduled[t]));estimate=nav
        for _ in range(40):
            goal=project(np.where(continuing,values,want*estimate/nav),frozen,estimate,m['issuers'],c);dollar=np.where(trad,goal-values,0);new=nav-fee*np.abs(dollar).sum()
            if abs(new-estimate)<nav*1e-10:break
            estimate=new
        else:raise RuntimeError('Known-close fee projection did not converge')
        pending=np.divide(dollar,mark,out=np.zeros(k),where=trad&(mark>0));decision=date
        if capture:
            for j in np.flatnonzero(np.abs(pending)*mark>nav*1e-10):orders.append({'decision_date':date,'j':int(j),'submitted_shares':float(pending[j]),'known_mark':float(mark[j])})
    return pd.DataFrame(rows),pd.DataFrame(trades),pd.DataFrame(holds),pd.DataFrame(orders)
