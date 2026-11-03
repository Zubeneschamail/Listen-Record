import hashlib
import json
from pathlib import Path
import queue
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from qwen_engine import QWEN_MODEL, REQUIRED_FILES, ASR_REVISION, ALIGNER_REVISION, complete_engine, engine_root


def make_engine(folder):
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in REQUIRED_FILES:
        target=folder/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b'fixture')
        files[name]=dict(bytes=7,sha256=hashlib.sha256(b'fixture').hexdigest())
    info=dict(protocol=1,asr_revision=ASR_REVISION,aligner_revision=ALIGNER_REVISION,files=files)
    (folder/'engine.json').write_text(json.dumps(info),encoding='utf-8')
    return info


class EngineIntegrityTests(unittest.TestCase):
    def test_new_profile_prefers_installed_component_only(self):
        from qwen_engine import default_model_label,QWEN_LABEL
        with patch('qwen_engine.engine_root',return_value=None):
            self.assertTrue(default_model_label().startswith('small '))
        with patch('qwen_engine.engine_root',return_value=Path('engine')):
            self.assertEqual(default_model_label(),QWEN_LABEL)

    def test_incomplete_or_corrupt_component_is_not_used(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);make_engine(folder)
            self.assertTrue(complete_engine(folder,verify_hashes=True))
            target=folder/REQUIRED_FILES[-1]
            target.write_bytes(b'changed')
            self.assertTrue(complete_engine(folder))
            self.assertFalse(complete_engine(folder,verify_hashes=True))
            target.unlink()
            self.assertFalse(complete_engine(folder))

    def test_wrong_revision_and_escaping_manifest_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);info=make_engine(folder)
            for values in (dict(info,asr_revision='unvalidated'),
                           dict(info,files=dict(info['files'],**{'../other':dict(bytes=1,sha256='')}))):
                (folder/'engine.json').write_text(json.dumps(values),encoding='utf-8')
                self.assertFalse(complete_engine(folder))

    def test_partial_bundled_component_can_use_complete_user_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            resources,data=Path(directory)/'resources',Path(directory)/'data'
            make_engine(resources/'qwen-engine');make_engine(data/'engines/qwen')
            (resources/'qwen-engine'/REQUIRED_FILES[0]).unlink()
            self.assertEqual(engine_root(resources,data),data/'engines/qwen')

    def test_optional_component_does_not_enter_whisper_download(self):
        from model_download import cached_model,download_worker
        pipe=Mock()
        with patch('qwen_engine.engine_root',return_value=Path('engine')),patch('huggingface_hub.snapshot_download') as download:
            self.assertEqual(cached_model(QWEN_MODEL),Path('engine'))
            download.assert_not_called()
        with patch('huggingface_hub.HfApi') as api:
            download_worker(QWEN_MODEL,pipe)
            api.assert_not_called();pipe.close.assert_called_once()
            self.assertEqual(pipe.send.call_args.args[0][0],'error')


class IntegratedRuntimeTests(unittest.TestCase):
    def test_factory_keeps_qwen_out_of_whisper_and_rejects_cpu(self):
        from model_runtime import load_model
        with patch('qwen_engine.engine_root',return_value=Path('engine')),patch('qwen_runtime.QwenModel') as qwen, \
                patch('model_runtime.WhisperModel') as whisper,patch('model_runtime.prepare_cuda') as cuda:
            self.assertIs(load_model(QWEN_MODEL),qwen.return_value)
            whisper.assert_not_called();cuda.assert_not_called()
            with self.assertRaisesRegex(RuntimeError,'CPU'):
                load_model(QWEN_MODEL,device='cpu')

    def test_gpu_check_uses_own_runtime_and_releases_it(self):
        from gpu_diagnostics import probe_gpu
        model=SimpleNamespace(model=SimpleNamespace(device='cuda'),close=Mock())
        with patch('model_runtime.load_model',return_value=model),patch('model_runtime.prepare_cuda') as cuda:
            self.assertIn('词时间戳',probe_gpu(QWEN_MODEL,Mock()))
            cuda.assert_not_called();model.close.assert_called_once()

    def test_startup_does_not_try_cpu_or_whisper_libraries_for_qwen(self):
        from startup_checks import check_local
        model=SimpleNamespace(model=SimpleNamespace(device='cuda'),close=Mock())
        reports={}
        with patch('model_download.cached_model',return_value=Path('engine')),patch('model_runtime.load_model',return_value=model) as load, \
                patch('model_runtime.prepare_cuda') as cuda:
            check_local(QWEN_MODEL,lambda key,state,text:reports.update({key:state}))
            self.assertEqual(reports,dict(model='success',gpu='success'))
            load.assert_called_once();cuda.assert_not_called();model.close.assert_called_once()

    def test_switch_and_failed_child_reload_release_previous_engine(self):
        from app import Transcriber
        for previous_name,closed in ((QWEN_MODEL,True),('small',False)):
            with self.subTest(previous_name=previous_name,closed=closed):
                engine=Transcriber(queue.Queue())
                old=SimpleNamespace(closed=closed,close=Mock())
                engine.model,engine.model_name=old,previous_name
                new=SimpleNamespace(model=SimpleNamespace(device='cuda'),close=Mock())
                def load(*args,**kwargs):
                    old.close.assert_called_once()
                    engine.stop()
                    return new
                with patch('app.load_model',side_effect=load) as factory,patch('app.pa.PyAudio') as audio:
                    audio.return_value.get_device_info_by_index.return_value = dict(
                        defaultSampleRate=16000,maxInputChannels=1)
                    engine.run([dict(index=1,source='system')],QWEN_MODEL,'zh')
                    factory.assert_called_once()
                engine.unload_model();new.close.assert_called_once()


if __name__=='__main__':
    unittest.main()
