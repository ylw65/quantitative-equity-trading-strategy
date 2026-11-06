"""Run frozen research stages; default remains training only."""
import argparse
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))
from pipeline import run
if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',nargs='?',default='train',choices=['train','validation','test'])
    run(parser.parse_args().stage)
