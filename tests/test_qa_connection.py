import queue
import threading
import unittest
from unittest.mock import Mock
from qa_connection import QAConnection, request_failure


class ConnectionTests(unittest.TestCase):
    def wait_result(self, connection, events):
        while True:
            events.get(timeout=3)
            if connection.state != 'checking':
                return connection.state

    def test_automatic_check_never_sends_prompt(self):
        events, probe = queue.Queue(), Mock(return_value='OK')
        connection = QAConnection(events, lambda: ('authenticated', 'configured'), probe)
        connection.check()
        self.assertEqual(self.wait_result(connection, events), 'authenticated')
        probe.assert_not_called()

    def test_manual_check_requires_successful_response(self):
        events, probe = queue.Queue(), Mock(return_value='OK')
        connection = QAConnection(events, lambda: ('authenticated', 'configured'), probe)
        connection.check(probe=True)
        self.assertEqual(self.wait_result(connection, events), 'verified')
        probe.assert_called_once()
        probe.side_effect = RuntimeError('401 Unauthorized')
        connection.check(probe=True)
        self.assertEqual(self.wait_result(connection, events), 'unauthenticated')

    def test_missing_key_blocks_probe(self):
        events, probe = queue.Queue(), Mock()
        connection = QAConnection(events, lambda: ('unauthenticated', 'configure key first'), probe)
        connection.check(probe=True)
        self.assertEqual(self.wait_result(connection, events), 'unauthenticated')
        probe.assert_not_called()

    def test_stale_check_cannot_overwrite_new_request_result(self):
        events, started, release, finished = queue.Queue(), threading.Event(), threading.Event(), threading.Event()
        def probe(cancel):
            started.set()
            release.wait(2)
            finished.set()
            return 'OK'
        connection = QAConnection(events, lambda: ('authenticated', 'configured'), probe)
        connection.check(probe=True)
        self.assertTrue(started.wait(2))
        old = connection.revision
        connection.record('401 Unauthorized')
        release.set()
        self.assertTrue(finished.wait(2))
        connection._publish(old, 'verified', 'stale')
        self.assertEqual(connection.state, 'unauthenticated')
        self.assertIn('额度', request_failure('429 quota')[1])
