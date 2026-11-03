import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tools import benchmark_transcription_accuracy as benchmark


class ASRBenchmarkTests(unittest.TestCase):
    def test_language_overrides_and_group_scores_use_actual_reference_totals(self):
        cases = [dict(name='symbols', audio='a.wav', reference='用C++。', group='mixed',
                      critical_terms=['C++'], language='zh'),
                 dict(name='plain', audio='b.wav', reference='we use the cache', group='en', language='zh')]
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            manifest = folder/'manifest.json'
            manifest.write_text(json.dumps(cases), encoding='utf-8')
            rows = [[(0, 1, '用C#。', 0)], [(0, 1, 'we use the cache', 0)]]*2
            with patch('sys.argv', ['benchmark', '--manifest', str(manifest), '--beams', '3',
                                    '--languages', 'zh', 'auto', '--output', str(folder/'out')]), \
                    patch.object(benchmark, 'load_model', return_value=SimpleNamespace(model=SimpleNamespace(device='cpu'))), \
                    patch.object(benchmark, 'decode_audio', return_value=np.ones(16000, dtype=np.float32)), \
                    patch.object(benchmark, 'decode_chunk', side_effect=rows) as decode, \
                    contextlib.redirect_stdout(io.StringIO()):
                benchmark.main()
            report = json.loads((folder/'out/report.json').read_text(encoding='utf-8'))
        self.assertEqual([call.args[3] for call in decode.call_args_list], ['zh', 'zh', None, None])
        self.assertEqual([row['language_policy'] for row in report['summaries']], ['zh', 'auto'])
        for summary in report['summaries']:
            self.assertEqual(summary['cases'], 2)
            self.assertEqual(summary['mixed_edits'], 1)
            self.assertEqual(summary['mixed_tokens'], 6)
            self.assertAlmostEqual(summary['mer'], 1/6)
            self.assertEqual(summary['term_recall'], 0)
            self.assertEqual(summary['groups']['mixed']['terms_total'], 1)
            self.assertEqual(summary['groups']['en']['mer'], 0)
            self.assertEqual(summary['cer'], 0)  # Symbol error is exposed separately.

    def test_unspoken_critical_term_is_rejected_before_loading_audio_or_model(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            manifest = folder/'manifest.json'
            manifest.write_text(json.dumps([dict(audio='missing.wav', reference='缓存', critical_terms=['Redis'])]),
                                encoding='utf-8')
            with patch('sys.argv', ['benchmark', '--manifest', str(manifest), '--output', str(folder/'out')]), \
                    patch.object(benchmark, 'load_model') as load, patch.object(benchmark, 'decode_audio') as audio:
                with self.assertRaises(ValueError):
                    benchmark.main()
                load.assert_not_called()
                audio.assert_not_called()


if __name__ == '__main__':
    unittest.main()
