"""Prepare only the pinned public model files for the optional offline engine."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def verified(path, item):
    if not path.is_file() or path.stat().st_size != item['bytes']:
        return False
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()==item['sha256']


def prepare(target, offline_source=None):
    metadata=json.loads((ROOT/'packaging/qwen-models.json').read_text(encoding='utf-8'))
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
    for model in metadata['models']:
        folder=target/model['folder'];folder.mkdir(parents=True,exist_ok=True)
        if all(verified(folder/name,item) for name,item in model['files'].items()):
            continue
        if offline_source:
            for name,item in model['files'].items():
                source=offline_source/model['folder']/name
                if not verified(source,item):
                    raise RuntimeError('Pinned model file missing or corrupt: '+str(source))
                shutil.copy2(source,folder/name)
        else:
            from huggingface_hub import snapshot_download
            snapshot_download(model['repo'],revision=model['revision'],local_dir=folder,
                allow_patterns=list(model['files']),token=False,max_workers=4)
        if not all(verified(folder/name,item) for name,item in model['files'].items()):
            raise RuntimeError('Pinned model verification failed: '+model['folder'])
    print(json.dumps(dict(models=len(metadata['models']),verified=True)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'build/qwen-models')
    parser.add_argument('--offline-source',type=Path)
    args=parser.parse_args()
    prepare(args.output,args.offline_source)
