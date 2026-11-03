"""Execute the standalone GPU engine without an installed Python dependency."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from qwen_engine import complete_engine
from qwen_runtime import QwenModel


def verify(engine,report):
    if not complete_engine(engine,verify_hashes=True):
        raise RuntimeError('Offline engine file validation failed')
    with QwenModel(engine) as model:
        segments,_=model.transcribe(np.zeros(16000,dtype=np.float32),language='zh')
        if list(segments) or model._sequence:
            raise RuntimeError('Silence reached the recognizer')
        pid=model.worker_pid
    if not model.closed:
        raise RuntimeError('Engine did not close')
    result=dict(ok=True,engine=str(engine),actual_pid=pid,gpu_warmup=True,
        offline_files_verified=True,silence_rejected_before_request=True)
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--engine',type=Path,default=ROOT/'build/qwen-engine')
    parser.add_argument('--report',type=Path,default=ROOT/'build/qwen-engine-smoke.json')
    args=parser.parse_args()
    verify(args.engine,args.report)
