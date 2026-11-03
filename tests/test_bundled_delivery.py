import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from bundled_delivery import bootstrap_delivery
from knowledge_embedding import Encoder
from reference_files import load_settings, save_settings
from test_knowledge_packages import write_package


class BundledDeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder = Encoder()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.resources = self.folder / 'resources'
        self.data = self.folder / 'data'
        self.data.mkdir()
        self.delivery = self.resources / 'customer-delivery'
        self.delivery.mkdir(parents=True)
        self.package = self.delivery / 'customer.wlkb'
        self.order = 'a' * 32
        self.write_delivery()

    def write_delivery(self):
        self.package.unlink(missing_ok=True)
        write_package(self.package, self.encoder, customer=self.order)
        self.manifest = {'protocol': 1, 'order_id': self.order, 'customer_id': self.order,
                         'package': 'customer.wlkb', 'sha256': hashlib.sha256(self.package.read_bytes()).hexdigest()}
        (self.delivery / 'delivery.json').write_text(json.dumps(self.manifest), encoding='utf-8')

    def test_first_start_copies_and_enables_once_and_respects_removal(self):
        self.assertTrue(bootstrap_delivery(self.resources, self.data))
        settings = load_settings(self.data / 'reference-files.json')
        self.assertTrue(settings['enabled'])
        managed = Path(settings['paths'][0])
        self.assertEqual(managed.read_bytes(), self.package.read_bytes())
        save_settings({'enabled': False, 'paths': []}, self.data / 'reference-files.json')
        self.assertFalse(bootstrap_delivery(self.resources, self.data))
        self.assertEqual(load_settings(self.data / 'reference-files.json')['paths'], [])

    def test_new_customer_replaces_previous_delivery_preserves_manual_sources(self):
        self.assertTrue(bootstrap_delivery(self.resources, self.data))
        original = load_settings(self.data / 'reference-files.json')
        manual = self.folder / 'manual.md'
        manual.write_text('manual')
        save_settings({'enabled': True, 'paths': original['paths'] + [str(manual)]}, self.data / 'reference-files.json')
        self.order = 'b' * 32
        self.write_delivery()
        self.assertTrue(bootstrap_delivery(self.resources, self.data))
        current = load_settings(self.data / 'reference-files.json')
        self.assertEqual(len(current['paths']), 2)
        self.assertIn(str(manual), current['paths'])
        self.assertNotIn(original['paths'][0], current['paths'])

    def test_corrupt_or_wrong_customer_package_does_not_change_settings(self):
        self.package.write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError, '校验失败'):
            bootstrap_delivery(self.resources, self.data)
        self.assertFalse((self.data / 'reference-files.json').exists())
        self.write_delivery()
        self.manifest['customer_id'] = 'other'
        (self.delivery / 'delivery.json').write_text(json.dumps(self.manifest), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '身份'):
            bootstrap_delivery(self.resources, self.data)
        self.assertFalse((self.data / 'reference-files.json').exists())

    def test_standard_program_without_delivery_does_nothing(self):
        self.assertFalse(bootstrap_delivery(self.folder / 'standard', self.data))


if __name__ == '__main__':
    unittest.main()
