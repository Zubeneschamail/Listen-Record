import base64
import json
from pathlib import Path
import tempfile
import unittest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.exceptions import InvalidSignature
from app_paths import migrate_legacy
from updates import verify_manifest, version_tuple

class DesktopDistributionTests(unittest.TestCase):
    def test_migration_keeps_new_data_and_originals(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder)/'old',Path(folder)/'new'
            (source/'recordings').mkdir(parents=True)
            (target/'recordings').mkdir(parents=True)
            (source/'recordings/a.txt').write_text('old')
            (target/'recordings/a.txt').write_text('new')
            (source/'recordings/b.jsonl').write_text('{}')
            migrate_legacy(source,target)
            migrate_legacy(source,target)
            self.assertEqual((target/'recordings/a.txt').read_text(),'new')
            self.assertEqual((target/'recordings/b.jsonl').read_text(),'{}')
            self.assertTrue((source/'recordings/b.jsonl').exists())

    def test_signed_metadata_rejects_tampering_and_foreign_downloads(self):
        key=Ed25519PrivateKey.generate()
        public=key.public_key().public_bytes_raw()
        value=dict(version='0.3.0',url='https://github.com/Zubeneschamail/Listen-Record/releases/download/v0.3.0/setup.exe',size=100,sha256='a'*64)
        raw=json.dumps(value).encode()
        signature=base64.b64encode(key.sign(raw))
        self.assertEqual(verify_manifest(raw,signature,public)['version'],'0.3.0')
        with self.assertRaises(InvalidSignature):
            verify_manifest(raw+b' ',signature,public)
        value['url']='https://example.com/setup.exe'
        raw=json.dumps(value).encode()
        with self.assertRaises(ValueError):
            verify_manifest(raw,base64.b64encode(key.sign(raw)),public)
        self.assertGreater(version_tuple('0.10.0'),version_tuple('0.9.0'))

    def test_download_integrity_and_partial_cleanup(self):
        import io
        import hashlib
        from unittest.mock import patch
        from updates import download_update
        payload = b'installer bytes'
        value=dict(version='0.3.0',url='https://github.com/Zubeneschamail/Listen-Record/releases/download/v0.3.0/setup.exe',
                   size=len(payload),sha256=hashlib.sha256(payload).hexdigest())
        with tempfile.TemporaryDirectory() as folder, patch('updates.DATA',Path(folder)):
            with patch('updates.urllib.request.urlopen',return_value=io.BytesIO(payload)):
                self.assertEqual(download_update(value).read_bytes(),payload)
            with patch('updates.urllib.request.urlopen',return_value=io.BytesIO(b'bad')):
                with self.assertRaises(ValueError):
                    download_update(value)
            self.assertFalse(list(Path(folder).rglob('*.part')))
