"""Stage the standalone engine with a complete install-time hash manifest."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from qwen_engine import ASR_REVISION, ALIGNER_REVISION, complete_engine


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def source_versions():
    return {name:sha(ROOT/name) for name in ('qwen_worker.py','qwen_alignment.py',
        'packaging/qwen-worker.spec','packaging/qwen_diagnostics.py','requirements-qwen-build.txt',
        'packaging/qwen-models.json','packaging/licenses/Apache-2.0.txt')}


def reusable(output):
    if not complete_engine(output,verify_hashes=True):
        return False
    previous=json.loads((output/'engine.json').read_text(encoding='utf-8')).get('source_sha256',{})
    return all(previous.get(name)==digest for name,digest in source_versions().items())


def bundle(runtime, models, output):
    build_root=(ROOT/'build').resolve()
    output=output.resolve()
    if output.parent!=build_root:
        raise ValueError('Engine staging must be a direct child of the workspace build directory')
    sources=source_versions()
    sources['WenluQwen.exe']=sha(runtime/'WenluQwen.exe')
    if complete_engine(output,verify_hashes=True):
        previous=json.loads((output/'engine.json').read_text(encoding='utf-8'))
        if previous.get('source_sha256')==sources:
            print(json.dumps(dict(reused=True,output=str(output))),flush=True)
            return
    stage=build_root/('qwen-stage-'+uuid.uuid4().hex)
    assert stage.resolve().parent==build_root
    shutil.copytree(runtime,stage)
    metadata=json.loads((ROOT/'packaging/qwen-models.json').read_text(encoding='utf-8'))
    for model in metadata['models']:
        folder=stage/'models'/model['folder'];folder.mkdir(parents=True)
        for name,item in model['files'].items():
            source=models/model['folder']/name
            if source.stat().st_size!=item['bytes'] or sha(source)!=item['sha256']:
                raise RuntimeError('Pinned model verification failed: '+str(source))
            shutil.copy2(source,folder/name)
    (stage/'NOTICE.txt').write_text(
        'Wenlu optional Chinese recognition engine\n'
        'Qwen3-ASR-0.6B and Qwen3-ForcedAligner-0.6B: Apache-2.0, Qwen team.\n'
        'Official model cards and provenance are included under models/.\n'
        'PyTorch, Transformers and other runtime licenses are retained in _internal/.\n'
        'No interview recordings, evaluation corpus or customer knowledge packages are included.\n',encoding='utf-8')
    shutil.copy2(ROOT/'packaging/licenses/Apache-2.0.txt',stage/'LICENSE-Qwen-models.txt')
    files={p.relative_to(stage).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p))
           for p in sorted(stage.rglob('*')) if p.is_file()}
    manifest=dict(protocol=1,asr_revision=ASR_REVISION,aligner_revision=ALIGNER_REVISION,
        model='Qwen3-ASR-0.6B',source_sha256=sources,files=files)
    (stage/'engine.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    if not complete_engine(stage,verify_hashes=True):
        raise RuntimeError('Staged engine failed verification')
    if output.exists():
        backup=build_root/('qwen-engine-previous-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8])
        assert output.parent==backup.resolve().parent==build_root
        output.rename(backup)
    stage.rename(output)
    print(json.dumps(dict(output=str(output),files=len(files),bytes=sum(v['bytes'] for v in files.values()))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--runtime',type=Path,default=ROOT/'build/qwen-dist/WenluQwen')
    parser.add_argument('--models',type=Path,default=ROOT/'build/qwen-models')
    parser.add_argument('--output',type=Path,default=ROOT/'build/qwen-engine')
    parser.add_argument('--check-cache',action='store_true')
    args=parser.parse_args()
    if args.check_cache:
        ready=reusable(args.output)
        print(json.dumps(dict(reusable=ready)),flush=True)
        raise SystemExit(0 if ready else 1)
    else:
        bundle(args.runtime,args.models,args.output)
