"""Create signed update metadata. Private key is supplied via a local file or CI secret."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from version import VERSION, REPOSITORY

parser=argparse.ArgumentParser()
parser.add_argument('installer', type=Path)
parser.add_argument('--key', type=Path)
args=parser.parse_args()
raw_key = args.key.read_bytes() if args.key else base64.b64decode(os.environ['WENLU_UPDATE_PRIVATE_KEY'])
key = Ed25519PrivateKey.from_private_bytes(raw_key)
expected = (Path(__file__).resolve().parents[1]/'assets/update-public.key').read_bytes()
assert key.public_key().public_bytes_raw() == expected, 'Signing key does not match embedded public key'
manifest = dict(version=VERSION, url=f'https://github.com/{REPOSITORY}/releases/download/v{VERSION}/{args.installer.name}',
                size=args.installer.stat().st_size, sha256=hashlib.file_digest(args.installer.open('rb'), 'sha256').hexdigest())
raw=json.dumps(manifest,ensure_ascii=False,sort_keys=True).encode('utf-8')
args.installer.with_name('update.json').write_bytes(raw)
args.installer.with_name('update.sig').write_bytes(base64.b64encode(key.sign(raw)))
print('Signed release metadata written')
