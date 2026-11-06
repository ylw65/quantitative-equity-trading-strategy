from pathlib import Path
import sys,json,importlib.metadata
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from strategy import config,load_market
from pipeline import load_frozen
def main():
    assert sys.prefix!=sys.base_prefix,'Use project .venv'
    assert 'include-system-site-packages = false' in (ROOT/'.venv/pyvenv.cfg').read_text()
    assert sys.version_info[:2]==(3,13),'Use Python3.13'
    c=config();model=load_frozen(c);m=load_market(c)
    installed={x.metadata['Name'].lower().replace('_','-'):x.version for x in importlib.metadata.distributions() if x.metadata['Name'].lower()!='pip'}
    expected={line.split('==')[0].lower().replace('_','-'):line.split('==')[1] for line in (ROOT/'environment-lock.txt').read_text().splitlines() if '==' in line}
    assert installed==expected,'Environment differs from exact lock'
    record={'passed':True,'version':c['version'],'securities':len(m['securities']),'max_loaded_date':str(m['dates'].max().date()),'data_directory':c['data_directory'],'dependency_count':len(installed),'backtest_executed':False,'ml_enabled':True,'frozen2023_threshold':model['quantiles']['0.5'],'test_rows':0}
    (ROOT/'docs').mkdir(exist_ok=True);(ROOT/'docs/environment_status.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))
if __name__=='__main__':main()
