import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

spec = importlib.util.spec_from_file_location('bundle_gpu', Path(__file__).resolve().parents[1] / 'packaging/bundle_gpu.py')
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


class GPUBundleTests(unittest.TestCase):
    def wheels(self, root, omit=None):
        directory = root / 'build/gpu-wheels'
        directory.mkdir(parents=True)
        for component, (package, version) in bundle.PACKAGES.items():
            path = directory / f'{package.replace("-", "_")}-{version}-py3-none-win_amd64.whl'
            with zipfile.ZipFile(path, 'w') as archive:
                for name in bundle.REQUIRED:
                    if name.startswith(component) and name != omit:
                        archive.writestr(f'nvidia/{component}/bin/{name}', b'test-dll')
                archive.writestr(package + '.dist-info/LICENSE.txt', 'NVIDIA license fixture')
                archive.writestr(f'nvidia/{component}/bin/../../escape.dll', 'ignored')

    def test_stages_dlls_and_licenses_without_installing_python_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.wheels(root)
            with patch.object(bundle, 'ROOT', root), patch.object(bundle.subprocess, 'run') as run:
                bundle.stage()
                run.assert_not_called()
            output = root / 'build/gpu-runtime'
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(len(list(output.rglob('*.dll'))), len(bundle.REQUIRED))
            self.assertEqual(len(list((output / 'licenses').rglob('LICENSE.txt'))), 2)
            self.assertIn('nvidia/cublas/bin/cublas64_12.dll', manifest['sha256'])
            self.assertFalse(list(root.rglob('escape.dll')))

    def test_missing_dependency_blocks_the_installer_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.wheels(root, omit='cudnn64_9.dll')
            with patch.object(bundle, 'ROOT', root):
                with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                    bundle.stage()
