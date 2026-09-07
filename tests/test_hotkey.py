import queue
import unittest
from unittest.mock import Mock

from hotkey import GlobalHotkey
from app import App


class HotkeyTests(unittest.TestCase):
    def test_native_registration_conflict_and_release(self):
        # Use a separate test combination so the live app's Ctrl+空格 is unaffected.
        first_events, second_events = queue.Queue(), queue.Queue()
        first = GlobalHotkey(first_events, modifiers=7, key=0x7B)
        second = GlobalHotkey(second_events, modifiers=7, key=0x7B)
        third = None
        try:
            first.start()
            self.assertEqual(first_events.get(timeout=3), ("hotkey_status", (True, "")))
            second.start()
            kind, (ok, reason) = second_events.get(timeout=3)
            self.assertEqual(kind, "hotkey_status")
            self.assertFalse(ok)
            self.assertTrue(reason)
            first.close()
            self.assertFalse(first.thread.is_alive())
            events = queue.Queue()
            third = GlobalHotkey(events, modifiers=7, key=0x7B)
            third.start()
            self.assertEqual(events.get(timeout=3), ("hotkey_status", (True, "")))
        finally:
            first.close()
            second.close()
            if third:
                third.close()

    def test_toggle_ignores_drain_and_close_but_can_cancel_loading(self):
        app = Mock(closing=False, busy=False)
        App.toggle_recording(app)
        app.start.assert_called_once()
        app.busy = True
        app.engine.stop_event.is_set.return_value = False
        App.toggle_recording(app)
        app.stop.assert_called_once()
        app.engine.stop_event.is_set.return_value = True
        App.toggle_recording(app)
        app.stop.assert_called_once()
        app.closing = True
        app.busy = False
        App.toggle_recording(app)
        app.start.assert_called_once()
