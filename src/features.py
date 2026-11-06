import numpy as np
import pandas as pd

def matrix(raw,name,m):
    return raw.pivot(index='date',columns='security_id',values=name).reindex(index=m['dates'],columns=m['securities'])

def features(m,f,raw,full,js):
    p,v=m['p'],m['v'];r=p.pct_change(fill_method=None);ret5=p/p.shift(5)-1;vol20=r.rolling(20).std();mv=(1+r.mean(axis=1)).cumprod();mr5=mv/mv.shift(5)-1;mr20=mv/mv.shift(20)-1
    high,low,close=[matrix(raw,name,m) for name in ['high','low','close']];span=high-low
    values={'ret1':r,'ret5':ret5,'ret20':p/p.shift(20)-1,'rv':v/v.shift(1).rolling(20).mean(),'clv':(2*close-high-low)/span.where(span>0),'range':span/close,'vol20':vol20,'vol_ratio':vol20/r.rolling(60).std(),'low_distance':p/m['low']-1,'relative5':ret5.sub(mr5,axis=0)}
    X=pd.DataFrame({name:value.to_numpy()[full,js] for name,value in values.items()})
    for name,value in [('market5',mr5),('market20',mr20),('breadth',ret5.le(-.05).where(ret5.notna()).mean(axis=1))]:X[name]=value.to_numpy()[full]
    shape=p.shape;latest=np.full(shape,-1,int);rv=np.full(shape,np.nan);er=rv.copy();score=rv.copy()
    for row in f[f.direction>0].itertuples():latest[int(row.t),int(row.j)]=int(row.t);rv[int(row.t),int(row.j)]=row.rv;er[int(row.t),int(row.j)]=row.ret;score[int(row.t),int(row.j)]=row.strength
    np.maximum.accumulate(latest,axis=0,out=latest)
    for a in [rv,er,score]:
        for t in range(1,len(p)):a[t]=np.where(np.isfinite(a[t]),a[t],a[t-1])
    s=latest[full,js];assert (s>=0).all();X['event_rv']=rv[full,js];X['event_ret']=er[full,js];X['event_score']=score[full,js];X['event_low_distance']=p.to_numpy()[full,js]/m['low'].to_numpy()[s,js]-1
    return X.replace([np.inf,-np.inf],np.nan)
