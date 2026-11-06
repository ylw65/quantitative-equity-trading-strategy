"""Risk projection and hard research-date boundary; no data or model dependencies."""
import numpy as np
import pandas as pd
LIMIT=pd.Timestamp("2023-12-31")

def assert_dates(dates):
    dates=pd.DatetimeIndex(dates)
    if len(dates) and dates.max()>LIMIT: raise ValueError('检测到test日期，立即停止')

def project(desired,frozen,nav,issuers,c):
    p=c['portfolio'];g=np.clip(desired.copy(),-p['security_cap']*nav,p['security_cap']*nav)
    fixed=np.bincount(issuers,weights=np.abs(frozen));free=np.bincount(issuers,weights=np.abs(g),minlength=len(fixed))
    g*=np.minimum(1,np.maximum(0,p['issuer_cap']*nav-fixed)/np.maximum(free,1e-300))[issuers]
    for sign,budget in [(1,p['long_budget']),(-1,p['short_budget'])]:
        mask=g*sign>0;used=np.maximum(sign*frozen,0).sum();s=np.abs(g[mask]).sum()
        if s:g[mask]*=min(1,max(0,budget*nav-used)/s)
    total=frozen+g;excess=abs(total.sum())-p['net_cap']*nav
    mask=g>0 if total.sum()>0 else g<0;s=np.abs(g[mask]).sum()
    if excess>0 and s:g[mask]*=max(0,1-excess/s)
    s=np.abs(g).sum()
    if s:g*=min(1,max(0,p['gross_cap']*nav-np.abs(frozen).sum())/s)
    return g
