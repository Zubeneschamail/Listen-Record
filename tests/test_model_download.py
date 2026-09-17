import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
from model_download import cached_model, download_worker


class ModelDownloadTests(TestCase):
    def test_complete_revision_without_main_ref_is_discovered(self):
        from huggingface_hub.errors import LocalEntryNotFoundError
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            snapshot = cache / 'models--Systran--faster-whisper-small/snapshots/revision'
            snapshot.mkdir(parents=True)
            for name in ('config.json', 'model.bin', 'tokenizer.json', 'vocabulary.txt'):
                (snapshot / name).write_text('complete')
            with patch('model_download.BUNDLED_MODELS', cache / 'missing'), \
                 patch('model_download.MODELS', cache), \
                 patch('huggingface_hub.snapshot_download', side_effect=LocalEntryNotFoundError('No main ref')):
                self.assertEqual(cached_model('small'), snapshot)

    def test_byte_progress_excludes_file_counts_and_network_transfer_duplicates(self):
        from unittest.mock import Mock
        from model_download import progress_text
        pipe = Mock()
        def fake_download(*args, **kwargs):
            progress = kwargs['tqdm_class']
            for desc, unit in [('Downloading bytes', 'B'), ('Fetching files', 'it'), ('Reconstructing', 'B')]:
                with progress(total=1024, unit=unit, desc=desc) as bar:
                    bar.update(512)
                    bar.update(512)
            return 'downloaded'
        with patch('model_download.cached_model', return_value=None), \
             patch('huggingface_hub.HfApi'), patch('huggingface_hub.snapshot_download', side_effect=fake_download), \
             patch('faster_whisper.WhisperModel'):
            download_worker('tiny', pipe)
        updates = [call.args[0][1] for call in pipe.send.call_args_list if call.args[0][0] == 'bytes']
        self.assertEqual(updates, [(512, 1024), (1024, 1024)])
        self.assertEqual(progress_text(1024**3, 2 * 1024**3), '50.0% · 1.00 / 2.00 GiB')
        self.assertNotIn('e+', progress_text(1541410000, 1617880000))

    def test_bundled_model_is_used_without_cache_or_network(self):
        with tempfile.TemporaryDirectory() as directory:
            bundled = Path(directory) / 'small'
            bundled.mkdir()
            for name in ('config.json', 'model.bin', 'tokenizer.json', 'vocabulary.txt'):
                (bundled / name).write_text('test model')
            with patch('model_download.BUNDLED_MODELS', Path(directory)), \
                 patch('huggingface_hub.snapshot_download') as fetch:
                self.assertEqual(cached_model('small'), bundled)
                fetch.assert_not_called()

    def test_incomplete_bundle_falls_back_to_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            bundled = Path(directory) / 'small'
            bundled.mkdir()
            (bundled / 'model.bin').write_text('incomplete')
            cached = Path(directory) / 'cached'
            cached.mkdir()
            for name in ('config.json', 'model.bin', 'tokenizer.json', 'vocabulary.txt'):
                (cached / name).write_text('test model')
            with patch('model_download.BUNDLED_MODELS', Path(directory)), \
                 patch('huggingface_hub.snapshot_download', return_value=str(cached)) as fetch:
                self.assertEqual(cached_model('small'), cached)
                self.assertTrue(fetch.call_args.kwargs['local_files_only'])

    def test_missing_model_does_not_download_when_starting_capture(self):
        from model_runtime import load_model
        with patch('model_download.cached_model', return_value=None), patch('model_runtime.WhisperModel') as model:
            with self.assertRaisesRegex(RuntimeError, '模型管理'):
                load_model('small', device='cpu')
            model.assert_not_called()

    def test_incomplete_current_cache_falls_back_to_legacy_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            current, legacy = Path(directory) / 'current', Path(directory) / 'legacy'
            current.mkdir()
            legacy.mkdir()
            for name in ('config.json', 'model.bin', 'tokenizer.json', 'vocabulary.txt'):
                (legacy / name).write_text('cached')
            (current / 'config.json').write_text('{}')
            with patch('model_download.MODELS', current), patch('model_download.legacy_models', return_value=legacy), \
                 patch('huggingface_hub.snapshot_download', side_effect=[str(current), str(legacy)]) as fetch:
                self.assertEqual(cached_model('small'), legacy)
                self.assertTrue(all(call.kwargs['local_files_only'] for call in fetch.call_args_list))

    def test_cached_worker_never_connects(self):
        from unittest.mock import Mock
        pipe = Mock()
        with patch('model_download.cached_model', return_value=Path('cached')), \
             patch('huggingface_hub.HfApi') as api, patch('faster_whisper.WhisperModel') as model:
            download_worker('small', pipe)
            api.assert_not_called()
            self.assertTrue(model.call_args.kwargs['local_files_only'])
            self.assertEqual(pipe.send.call_args.args[0][0], 'done')

    def test_connection_failure_is_reported(self):
        from unittest.mock import Mock
        pipe = Mock()
        with patch('model_download.cached_model', return_value=None), \
             patch('huggingface_hub.HfApi') as api:
            api.return_value.model_info.side_effect = TimeoutError('timeout')
            download_worker('small', pipe)
            self.assertEqual(pipe.send.call_args.args[0][0], 'error')
            pipe.close.assert_called_once()
