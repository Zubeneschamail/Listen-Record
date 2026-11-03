import io
import json
import queue
import threading
import tkinter as tk
import unittest
from unittest.mock import patch

from PIL import Image

from app import App
from clipboard_watch import ClipboardItem
from screen_capture import ScreenCapture, capture_screen, foreground_monitor_bounds


def item():
    output = io.BytesIO()
    Image.new('RGB',(160,90),'blue').save(output,format='PNG')
    return ClipboardItem('【屏幕截图】当前屏幕中的问题',output.getvalue())


class ScreenCaptureTests(unittest.TestCase):
    def test_snapshot_uses_only_foreground_monitor_and_preserves_negative_coordinates(self):
        bounds=(-1920,0,0,1080)
        with patch('screen_capture.foreground_monitor_bounds',return_value=bounds), \
                patch('screen_capture.ImageGrab.grab',return_value=Image.new('RGB',(1920,1080),'white')) as grab:
            result=capture_screen()
        grab.assert_called_once_with(bbox=bounds,all_screens=True)
        self.assertIn('屏幕截图',result.text)
        self.assertNotIn('剪贴板',result.text)
        with Image.open(io.BytesIO(result.image)) as image:
            self.assertEqual(image.size,(1920,1080))

    def test_no_capture_before_request_and_inflight_or_cancelled_requests_are_not_uploaded(self):
        events=queue.Queue()
        begin,release=threading.Event(),threading.Event()
        calls=[]
        def capture():
            calls.append(True)
            begin.set()
            release.wait(3)
            return item()
        controller=ScreenCapture(events,capture=capture)
        self.assertFalse(calls)
        self.assertTrue(controller.request())
        self.assertTrue(begin.wait(1))
        self.assertFalse(controller.request())
        controller.cancel()
        release.set()
        with self.assertRaises(queue.Empty):
            events.get(timeout=.2)
        self.assertEqual(len(calls),1)

    def test_capture_error_is_reported_without_an_image(self):
        events=queue.Queue()
        def broken():
            raise OSError('设备不可用')
        controller=ScreenCapture(events,capture=broken)
        controller.request()
        kind,(generation,picture,error)=events.get(timeout=2)
        self.assertEqual(kind,'screen_capture')
        self.assertIsNone(picture)
        self.assertIn('设备不可用',error)


class ScreenQuestionAppTests(unittest.TestCase):
    def setUp(self):
        self.patches=[patch('app.GlobalHotkey'),patch('app.App.start_tray'),
                      patch('qa_connection.QAConnection.check'),patch('app.ClipboardWatcher')]
        for p in self.patches:
            p.start()
        self.root=tk.Tk()
        self.app=App(self.root)
        self.app.qa_connection.state='authenticated'
        self.app.qa_settings['provider']='compatible'

    def tearDown(self):
        self.app.screen_capture.cancel()
        self.app.qa.set_enabled(False)
        self.app.qa_connection.close()
        for timer in self.root.tk.call('after','info'):
            self.root.after_cancel(timer)
        self.root.destroy()
        for p in reversed(self.patches):
            p.stop()

    def finish(self):
        while True:
            kind,value=self.app.events.get(timeout=3)
            if kind=='qa':
                self.app.handle_qa(value)
                if value[1]=='done':
                    return

    def test_explicit_screen_question_works_without_auto_and_omits_unrelated_audio(self):
        received=[]
        self.app.qa.stream_runner=None
        self.app.qa.runner=lambda prompt,cancel,image=None: received.append((prompt,image)) or '屏幕题答案'
        self.app.rows=[{'text':'无关的录音内容不应附带'}]
        self.assertFalse(self.app.auto_qa.get())
        picture=item()
        self.app.handle_screen_capture((self.app.screen_capture.generation,picture,''))
        self.finish()
        self.assertEqual(received[0][1],picture.image)
        self.assertNotIn('无关的录音内容不应附带',received[0][0])
        self.assertEqual(self.app.qa.source,'screen')
        self.assertEqual(self.app.qa_answer,'屏幕题答案')
        self.assertEqual(len(self.app.qa_display_images),1)
        self.assertFalse(self.app.auto_qa.get())

    def test_clear_or_disable_discards_a_late_snapshot(self):
        self.app.qa.runner=lambda *args,**kwargs: self.fail('must not upload')
        generation=self.app.screen_capture.generation
        self.app.clear()
        self.app.handle_screen_capture((generation,item(),''))
        self.assertIsNone(self.app.screen_pending_item)
        self.assertFalse(self.app.qa.active)
        self.app.screen_capture_enabled.set(False)
        self.app.apply_screen_hotkey()
        self.app.handle_screen_capture((self.app.screen_capture.generation,item(),''))
        self.assertFalse(self.app.qa.active)

    def test_screen_question_does_not_attach_live_transcription_when_auto_is_on(self):
        prompts=[]
        self.app.auto_qa.set(True)
        self.app.rows=[{'text':'录音中的内容不可附加到屏幕提问'}]
        self.app.qa.stream_runner=None
        self.app.qa.runner=lambda prompt,cancel,image=None: prompts.append(prompt) or '答案'
        self.app.handle_screen_capture((self.app.screen_capture.generation,item(),''))
        self.finish()
        self.assertNotIn('录音中的内容不可附加到屏幕提问',prompts[0])

    def test_missing_connection_prevents_capture(self):
        self.app.qa_connection.state='unauthenticated'
        with patch.object(self.app.screen_capture,'request') as capture:
            self.app.request_screen_question()
        capture.assert_not_called()
        self.assertIn('检查答疑模型',self.app.qa_status.get())

    def test_closed_answer_panel_waits_for_preflight_then_sends_snapshot(self):
        received=[]
        self.app.qa.stream_runner=None
        self.app.qa.runner=lambda prompt,cancel,image=None: received.append(image) or '答案'
        self.app.qa_enabled.set(False)
        self.app.toggle_qa()
        def checking():
            self.app.qa_connection.state='checking'
        with patch.object(self.app.qa_connection,'check',side_effect=checking):
            self.app.handle_screen_capture((self.app.screen_capture.generation,item(),''))
        self.assertIsNotNone(self.app.screen_pending_item)
        self.assertFalse(received)
        self.app.qa_connection.state='authenticated'
        self.app.send_pending_screen()
        self.finish()
        self.assertEqual(len(received),1)
        self.assertTrue(self.app.qa_enabled.get())

    def test_screen_hotkey_settings_are_drafted_and_apply_only_on_save(self):
        original=self.app.screen_capture_enabled.get()
        self.app.toggle_settings()
        self.app.screen_capture_enabled.set(not original)
        self.app.hide_settings()
        self.assertEqual(self.app.screen_capture_enabled.get(),original)
        self.app.toggle_settings()
        self.app.screen_capture_enabled.set(not original)
        with patch('app.save_desktop') as save, patch('reference_settings.save_settings'), \
                patch.object(self.app,'apply_screen_hotkey') as apply:
            self.app.settings_save_button.invoke()
        self.assertEqual(save.call_args.args[0]['screen_capture_hotkey'],not original)
        apply.assert_called_once()
