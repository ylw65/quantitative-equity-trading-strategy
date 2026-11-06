"""Load the research period directly from the repository-owned dataset."""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import exchange_calendars as xcals

def dataset_directory(root,c):
    directory=(Path(root)/c['data_directory']).resolve()
    if not directory.is_relative_to(Path(root).resolve()):raise ValueError('Data must be inside the repository')
    for filename in ['selected_equity_daily.parquet','selected_security_master.parquet','dataset_metadata.json']:
        if not (directory/filename).is_file():raise ValueError(f'Missing dataset file: {filename}')
    return directory

def check_window(c):
    start=pd.Timestamp(c['allowed_start']);end=pd.Timestamp(c['allowed_end'])
    if end>pd.Timestamp('2025-12-31'):
        raise ValueError('Dates after the authorised test period are forbidden')
    if end>pd.Timestamp('2023-12-31') and not c.get('test_enabled',False):
        raise ValueError('2024–2025 test is forbidden without explicit test mode, before any data read')
    if start<pd.Timestamp('2016-01-01') or start>end:raise ValueError('Invalid research window')
    if end>pd.Timestamp('2022-12-31') and not (c.get('validation_enabled',False) or c.get('test_enabled',False)):
        raise ValueError('2023 requires explicit validation authorization')
    return start,end

def read_daily(root,c):
    start,end=check_window(c)
    directory=dataset_directory(root,c)
    expr=(ds.field('date')>=start)&(ds.field('date')<=end)
    return ds.dataset(directory/'selected_equity_daily.parquet').to_table(
        columns=['date','security_id','high','low','close','adj_close','volume'],filter=expr).to_pandas()

def load_market(root,c,raw=None):
    start,end=check_window(c)
    directory=dataset_directory(root,c)
    daily=read_daily(root,c) if raw is None else raw.copy()
    master=ds.dataset(directory/'selected_security_master.parquet').to_table(columns=['security_id','cik']).to_pandas().sort_values('security_id').reset_index(drop=True)
    if daily.empty or daily.date.min()<start or daily.date.max()>end:raise ValueError('Missing/out-of-range research data')
    if daily.duplicated(['date','security_id']).any() or master.security_id.duplicated().any():raise ValueError('Duplicate data keys')
    if not (daily.adj_close.gt(0)&daily.volume.ge(0)).all():raise ValueError('Invalid price or volume')
    if not daily.security_id.isin(master.security_id).all():raise ValueError('Unknown security ID')
    securities=master.security_id.to_list()
    dates=xcals.get_calendar('XNYS').sessions_in_range(daily.date.min(),daily.date.max()).tz_localize(None)
    if not daily.date.isin(dates).all():raise ValueError('Non-session observation')
    p=daily.pivot(index='date',columns='security_id',values='adj_close').reindex(index=dates,columns=securities)
    # Apply the same adjustment factor as the mark price to the event-day low.
    daily['adj_low']=daily['low']*daily['adj_close']/daily['close']
    low=daily.pivot(index='date',columns='security_id',values='adj_low').reindex(index=dates,columns=securities)
    v=daily.pivot(index='date',columns='security_id',values='volume').reindex(index=dates,columns=securities)
    tradable=p.notna().to_numpy()&v.gt(0).to_numpy()
    issuer_key=master.cik.fillna('').where(master.cik.fillna('').ne(''),master.security_id)
    issuers=pd.factorize(issuer_key)[0]
    return {'dates':dates,'p':p,'low':low,'v':v,'securities':securities,'issuers':issuers,'tradable':tradable,'allowed':tradable.copy()}
