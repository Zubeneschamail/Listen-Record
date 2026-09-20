import base64
import io
import json
from pathlib import Path
import queue
import threading
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx
from PIL import Image

from app import App
from clipboard_watch import ClipboardItem, ClipboardWatcher, image_item
from qa_worker import QAWorker, make_prompt
from qa_provider import APIProvider


class Source:
    def __init__(self):
        self.number = 1
        self.item = ClipboardItem('开启前的旧内容')
        self.own = False
        self.busy = 0

    def sequence(self):
        return self.number

    def owned_by_app(self):
        return self.own

    def read(self):
        if self.busy:
            self.busy -= 1
            raise OSError('busy')
        return self.item

    def copy(self, item, own=False):
        self.item, self.own = item, own
        self.number += 1


class ClipboardWatcherTests(unittest.TestCase):
    def setUp(self):
        self.events, self.source = queue.Queue(), Source()
        self.watcher = ClipboardWatcher(self.events, self.source)
        self.watcher.start()

    def tearDown(self):
        self.watcher.stop()

    def test_only_new_copies_and_deduplication_and_self_copy(self):
        with self.assertRaises(queue.Empty):
            self.events.get(timeout=.5)
        self.source.copy(ClipboardItem('新问题'))
        self.assertEqual(self.events.get(timeout=2)[1][1].text, '新问题')
        self.source.copy(ClipboardItem('新问题'))
        with self.assertRaises(queue.Empty):
            self.events.get(timeout=.7)
        self.source.copy(ClipboardItem('来自闻录的答案'), own=True)
        with self.assertRaises(queue.Empty):
            self.events.get(timeout=.7)
        self.source.copy(ClipboardItem('下一个问题'))
        self.assertEqual(self.events.get(timeout=2)[1][1].text, '下一个问题')

    def test_busy_clipboard_retries_and_rejected_copy_can_be_repeated(self):
        self.source.busy = 2
        self.source.copy(ClipboardItem('稍后可读取'))
        self.assertEqual(self.events.get(timeout=3)[1][1].text, '稍后可读取')
        self.watcher.allow_repeat()
        self.source.copy(ClipboardItem('稍后可读取'))
        self.assertEqual(self.events.get(timeout=2)[1][1].text, '稍后可读取')

    def test_changed_during_read_is_not_sent_under_old_sequence(self):
        original = self.source.read
        def racing_read():
            old = original()
            self.source.copy(ClipboardItem('最新内容'))
            self.source.read = original
            return old
        self.source.read = racing_read
        self.source.copy(ClipboardItem('已经过时'))
        self.assertEqual(self.events.get(timeout=3)[1][1].text, '最新内容')
        with self.assertRaises(queue.Empty):
            self.events.get(timeout=.5)

    def test_stop_and_restart_ignore_previous_clipboard(self):
        old_generation = self.watcher.generation
        self.watcher.stop()
        self.source.copy(ClipboardItem('关闭期间复制'))
        self.watcher.start()
        self.assertNotEqual(old_generation, self.watcher.generation)
        with self.assertRaises(queue.Empty):
            self.events.get(timeout=.6)


class ImageRequestTests(unittest.TestCase):
    def test_normalize_image_and_reject_excessive_dimensions(self):
        item = image_item(Image.new('RGBA', (5000, 100), 'red'))
        self.assertIn('4096×82', item.text)
        with Image.open(io.BytesIO(item.image)) as actual:
            self.assertEqual(actual.size, (4096, 82))
        with self.assertRaises(ValueError):
            image_item(Mock(width=6000, height=6000))

    def test_api_sends_actual_image_and_keeps_system_prompt_separate(self):
        item = image_item(Image.new('RGB', (20, 10), 'red'))
        def handler(request):
            messages = json.loads(request.content)['messages']
            self.assertEqual(messages[0]['role'], 'system')
            content = messages[1]['content']
            self.assertEqual(json.loads(content[0]['text'])['当前问题'], item.text)
            self.assertEqual(base64.b64decode(content[1]['image_url']['url'].split(',', 1)[1]), item.image)
            return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"红色"},"finish_reason":"stop"}]}\n\n')
        backend = APIProvider('compatible', dict(base_url='https://example.com/v1',
            api_key='test', model='vision'), httpx.MockTransport(handler))
        self.assertEqual(backend.run(make_prompt('', item.text), threading.Event(), image=item.image), '红色')
        backend.provider = 'deepseek'
        backend.profile['model'] = 'deepseek-flash'
        self.assertEqual(backend.run(make_prompt('', item.text), threading.Event(), image=item.image), '红色')
        backend.profile['model'] = 'unsupported-model'
        with self.assertRaisesRegex(RuntimeError, '请选择|选择 deepseek-flash'):
            backend.run(make_prompt('', item.text), threading.Event(), image=item.image)

    def test_qa_worker_delivers_image_without_leaking_it_into_next_text_request(self):
        events, received = queue.Queue(), []
        def runner(prompt, cancel, image=None):
            received.append(image)
            return '已回答'
        qa = QAWorker(events, runner)
        qa.set_enabled(True)
        for image in (b'png-data', None):
            qa.ask('问题', [], image=image)
            while events.get(timeout=3)[1][1] != 'done':
                pass
        self.assertEqual(received, [b'png-data', None])


class ClipboardAppTests(unittest.TestCase):
    def setUp(self):
        self.patches = [patch('app.GlobalHotkey'), patch('app.App.start_tray'),
                       patch('qa_connection.QAConnection.check'),
                       patch('app.ClipboardWatcher')]
        for p in self.patches:
            p.start()
        self.root = tk.Tk()
        self.app = App(self.root)
        self.app.qa_connection.state = 'authenticated'
        self.app.qa_settings['provider'] = 'compatible'

    def tearDown(self):
        self.app.stop_clipboard()
        self.app.qa.set_enabled(False)
        self.app.qa_connection.close()
        self.app.hotkey.close()
        for timer in self.root.tk.call('after', 'info'):
            self.root.after_cancel(timer)
        self.root.destroy()
        for p in reversed(self.patches):
            p.stop()

    def enable(self):
        if not self.app.auto_qa.get():
            self.app.auto_qa.set(True)
            self.app.toggle_auto_qa()

    def enqueue(self, item):
        self.app.handle_clipboard((self.app.clipboard_watcher.generation, item, ''))

    def finish(self):
        while True:
            kind, value = self.app.events.get(timeout=3)
            if kind == 'qa':
                self.app.handle_qa(value)
                if value[1] == 'done':
                    return

    def test_queue_waits_preserves_answers_and_passes_image(self):
        self.assertFalse(self.app.auto_qa.get())
        self.enable()
        received = []
        self.app.qa.stream_runner = None
        self.app.qa.runner = lambda prompt, cancel, image=None: received.append((prompt, image)) or '答案'
        self.app.rows = [{'text': '不应发送的语音'}]
        self.app.qa.active = True
        self.enqueue(ClipboardItem('文字问题'))
        self.enqueue(ClipboardItem('图片问题', b'image'))
        self.app.send_pending_clipboard()
        self.assertEqual(len(self.app.clipboard_pending), 2)
        self.app.qa.active = False
        self.app.send_pending_clipboard()
        self.finish()
        self.app.send_pending_clipboard()
        self.finish()
        self.assertEqual([entry[1] for entry in received], [None, b'image'])
        self.assertNotIn('不应发送的语音', received[0][0])
        self.assertEqual(self.app.qa_display_history, [('文字问题', '答案')])
        self.assertEqual(self.app.qa_answer, '答案')

    def test_image_then_short_reply_is_one_role_preserving_conversation(self):
        self.enable()
        self.app.rows = [{'text': '不相关录音不要发给模型'}]
        requests = []
        offered = '这是两数之和问题，需要我写 Python 代码吗？'
        description = '题目：nums=[2,7,11,15]，target=9，返回两个下标。'
        def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            answer = description if body['model'] == 'deepseek-flash' else offered if len(requests) == 2 else 'def two_sum(nums, target): ...'
            return httpx.Response(200, text='data: '+json.dumps({'choices': [
                {'delta': {'content': answer}, 'finish_reason': 'stop'}]})+'\n\n')
        self.app.qa_settings['provider'] = 'deepseek'
        self.app.qa_settings['profiles']['deepseek'] = {'model': 'deepseek-v4-pro'}
        self.app.qa.stream_runner = APIProvider('deepseek', dict(base_url='https://example.com',
            model='deepseek-v4-pro', api_key='fake'), httpx.MockTransport(handler)).run
        self.enqueue(ClipboardItem('【剪贴板图片】请分析', b'image'))
        self.app.send_pending_clipboard()
        self.finish()
        self.assertEqual(self.app.qa_image_contexts, {0: description})
        self.enqueue(ClipboardItem('要的'))
        self.app.send_pending_clipboard()
        self.finish()
        self.assertEqual([b['model'] for b in requests], ['deepseek-flash', 'deepseek-v4-pro', 'deepseek-v4-pro'])
        messages = requests[-1]['messages']
        self.assertEqual([m['role'] for m in messages], ['system', 'user', 'assistant', 'user'])
        self.assertIn(description, messages[1]['content'])
        self.assertEqual(messages[2]['content'], offered)
        self.assertEqual(json.loads(messages[-1]['content'])['当前问题'], '要的')
        self.assertNotIn(offered, messages[-1]['content'])
        self.assertNotIn('不相关录音', json.dumps(messages, ensure_ascii=False))
        self.assertEqual(self.app.qa_answer, 'def two_sum(nums, target): ...')
        self.app.clear()
        self.enqueue(ClipboardItem('新问题'))
        self.app.send_pending_clipboard()
        self.finish()
        self.assertEqual([m['role'] for m in requests[-1]['messages']], ['system', 'user'])
        self.assertNotIn('target=9', json.dumps(requests[-1], ensure_ascii=False))
        self.assertFalse(self.app.qa_image_contexts)

    def test_manual_repeated_followup_uses_latest_answer_without_auto(self):
        self.app.open_qa()
        self.app.rows = [{'text': '所选段落'}]
        prompts = []
        self.app.qa.stream_runner = None
        self.app.qa.runner = lambda prompt, cancel: prompts.append(prompt) or f'第{len(prompts)}步'
        for _ in range(3):
            self.app.select_question('继续', [0])
            self.finish()
        self.assertFalse(self.app.auto_qa.get())
        self.assertEqual(len(prompts), 3)
        session = json.loads(prompts[-1].split('\n', 1)[1])['会话资料']
        self.assertIn('第2步', str(session['历史问答']))
        self.assertIn('所选段落', str(session['全部转写']))
        self.assertEqual(self.app.qa_display_history, [('继续', '第1步'), ('继续', '第2步')])

    def test_inline_images_survive_streaming_followups_and_provider_change(self):
        self.enable()
        self.root.geometry('640x440')
        self.root.update()
        received = []
        def stream(prompt, cancel, partial, **kwargs):
            received.append(kwargs.get('image'))
            partial('正在分析')
            return '分析完成'
        self.app.qa.stream_runner = stream
        first = image_item(Image.new('RGB', (800, 400), 'blue'))
        self.enqueue(first)
        self.app.send_pending_clipboard()
        names = self.app.qa_text.image_names()
        self.assertEqual(len(names), 1)  # Visible before the model has answered.
        self.finish()
        self.assertEqual(self.app.qa_text.image_names(), names)
        self.assertNotIn('请识别图片内容', self.app.qa_text.get('1.0', 'end-1c'))
        self.assertIn('分析完成', self.app.qa_text.get('1.0', 'end-1c'))
        self.enqueue(ClipboardItem('继续'))
        self.app.send_pending_clipboard()
        self.finish()
        self.assertEqual(len(self.app.qa_text.image_names()), 1)
        second = image_item(Image.new('RGB', (800, 400), 'red'))
        self.assertEqual(first.text, second.text)
        self.enqueue(second)
        self.app.send_pending_clipboard()
        self.finish()
        self.assertEqual(received, [first.image, None, second.image])
        self.assertEqual(set(self.app.qa_display_images), {0, 2})
        self.assertEqual(self.app.qa_display_images[0].source.getpixel((0, 0)), (0, 0, 255))
        self.assertEqual(self.app.qa_display_images[2].source.getpixel((0, 0)), (255, 0, 0))
        self.app.configure_qa_provider()
        self.assertEqual(len(self.app.qa_text.image_names()), 2)
        self.app.clear()
        self.assertFalse(self.app.qa_display_images)
        self.assertFalse(self.app.qa_text.image_names())

    def test_image_resize_preserves_editor_draft_and_question_indices(self):
        self.enable()
        self.root.geometry('720x480')
        self.root.update()
        self.app.qa.stream_runner = None
        self.app.qa.runner = lambda prompt, cancel, **kwargs: '图片答案'
        self.enqueue(image_item(Image.new('RGB', (1200, 800), 'blue')))
        self.app.send_pending_clipboard()
        self.finish()
        preview = self.app.qa_display_images[0]
        wide = preview.photo.width()
        self.enqueue(ClipboardItem('解释一下'))
        self.app.send_pending_clipboard()
        self.finish()
        start = self.app.qa_text.tag_ranges('question_turn:1')[0]
        self.app.qa_text.see(start)
        self.root.update()
        x, y, _, _ = self.app.qa_text.bbox(start)
        self.app.edit_question(SimpleNamespace(x=x+2, y=y+2))
        editor = self.app.question_editor
        editor.insert('end', '草稿')
        self.root.geometry('420x480')
        self.root.update()
        self.assertLess(preview.photo.width(), wide)
        self.assertLessEqual(preview.photo.width(), self.app.qa_text.winfo_width()-48)
        self.assertIs(self.app.question_editor, editor)
        self.assertEqual(editor.get('1.0', 'end-1c'), '解释一下草稿')
        self.assertEqual(len(self.app.qa_text.image_names()), 1)
        self.app.cancel_question_edit()
        self.assertEqual(self.app.qa_text.get(*self.app.qa_text.tag_ranges('question_turn:1')).strip(), '解释一下')

    def test_partial_and_cancelled_image_context_never_enter_next_conversation(self):
        self.enable()
        generation = self.app.qa.generation
        self.app.handle_qa((generation, 'image_context', '尚未完成的识别'))
        self.app.handle_qa((generation, 'partial', ('失败的问题', '不完整答案')))
        self.assertEqual(self.app.session_context_snapshot()[1], [])
        self.app.qa.reset()
        self.app.handle_qa((generation, 'answer', ('失败的问题', '迟到答案')))
        self.app.handle_qa((self.app.qa.generation, 'answer', ('新问题', '新答案')))
        self.assertEqual(self.app.session_context_snapshot()[1], [('新问题', '新答案')])
        self.assertFalse(self.app.qa_image_contexts)

    def test_disable_cancels_active_request_and_discards_late_events(self):
        self.enable()
        self.enqueue(ClipboardItem('问题'))
        cancel = self.app.qa.cancel
        self.app.qa.active = True
        self.app.clipboard_request_generation = self.app.qa.generation
        old_generation = self.app.qa.generation
        self.app.auto_button.invoke()
        self.assertFalse(self.app.auto_qa.get())
        self.assertTrue(cancel.is_set())
        self.assertFalse(self.app.clipboard_pending)
        self.enqueue(ClipboardItem('迟到内容'))
        self.app.handle_qa((old_generation, 'answer', ('旧问题', '旧答案')))
        self.assertFalse(self.app.clipboard_pending)
        self.assertEqual(self.app.qa_answer, '')

    def test_image_phases_are_status_only_and_late_phases_are_ignored(self):
        self.enable()
        generation = self.app.qa.generation
        self.app.clipboard_request_generation = generation
        self.app.qa_question = '图片问题'
        for stage in ('Flash 正在读图…', '读图完成，Pro 正在分析…'):
            self.app.handle_qa((generation, 'phase', stage))
            self.assertEqual(self.app.qa_status.get(), stage)
            self.assertEqual(self.app.clipboard_status.get(), stage)
            self.assertFalse(self.app.qa_history)
            self.assertEqual(self.app.qa_answer, '')
        self.app.handle_qa((generation, 'answer', ('图片问题', 'Pro 最终答案')))
        self.app.handle_qa((generation, 'done', None))
        self.assertEqual(self.app.qa_history, [('图片问题', 'Pro 最终答案')])
        self.assertEqual(self.app.clipboard_status.get(), '监听中 · 0 条待发送')
        self.app.stop_clipboard()
        self.app.qa.reset()
        self.app.handle_qa((generation, 'phase', '过期阶段'))
        self.assertNotEqual(self.app.qa_status.get(), '过期阶段')

    def test_auto_ai_close_and_provider_switch_keep_monitoring(self):
        self.enable()
        self.app.auto_qa.set(True)
        self.app.toggle_auto_qa()
        self.assertTrue(self.app.auto_qa.get())
        self.assertTrue(self.app.auto_qa.get())
        self.enqueue(ClipboardItem('后台问题'))
        self.app.qa_enabled.set(False)
        self.app.toggle_qa()
        self.assertTrue(self.app.auto_qa.get())
        self.assertTrue(self.app.qa.enabled)
        self.app.configure_qa_provider()
        self.assertTrue(self.app.auto_qa.get())
        self.assertEqual(len(self.app.clipboard_pending), 1)
        self.app.qa_connection.state = 'authenticated'
        self.app.qa.stream_runner = None
        self.app.qa.runner = lambda prompt, cancel: '后台答案'
        self.app.send_pending_clipboard()
        self.finish()
        self.assertEqual(self.app.qa_history, [('后台问题', '后台答案')])
        self.assertFalse(self.app.qa_enabled.get())
        self.app.open_qa()
        self.assertTrue(self.app.qa_enabled.get())
        self.assertIn('后台答案', self.app.qa_text.get('1.0', 'end-1c'))

    def test_auto_and_clipboard_share_history_without_losing_voice_queue(self):
        self.app.open_qa()
        self.app.auto_qa.set(True)
        self.app.toggle_auto_qa()
        self.app.rows = [{'text': '录音背景'}]
        self.app.qa.feed('语音问题是什么？')
        self.app.qa.last_input -= 2
        self.enable()
        self.assertTrue(self.app.auto_qa.get())
        received = []
        self.app.qa.stream_runner = None
        self.app.qa.runner = lambda prompt, cancel, image=None: received.append((prompt, image)) or '答案'
        self.enqueue(ClipboardItem('图片问题', b'image'))
        self.enqueue(ClipboardItem('继续'))
        self.app.send_pending_clipboard()
        self.app.qa.feed('再补充一个问题？')
        self.app.qa.last_input -= 2
        self.finish()
        self.assertNotIn('录音背景', received[0][0])
        self.assertEqual(self.app.qa.pending, ['语音问题是什么？', '再补充一个问题？'])
        self.app.poll()  # Ready voice gets a turn before the next clipboard copy.
        self.assertEqual(self.app.qa.source, 'auto')
        self.finish()
        self.app.send_pending_clipboard()
        self.finish()
        self.assertEqual([entry[1] for entry in received], [b'image', None, None])
        self.assertIn('录音背景', received[1][0])
        self.assertIn('图片问题', received[1][0])
        self.assertIn('再补充一个问题', received[2][0])
        self.assertNotIn('录音背景', received[2][0])
        self.assertEqual(len(self.app.qa_history), 3)

    def test_panel_toggles_do_not_cancel_inflight_clipboard(self):
        self.enable()
        release = threading.Event()
        self.app.qa.stream_runner = None
        self.app.qa.runner = lambda prompt, cancel: release.wait(3) and '完成'
        self.enqueue(ClipboardItem('问题'))
        self.app.send_pending_clipboard()
        cancellation = self.app.qa.cancel
        generation = self.app.qa.generation
        try:
            self.app.qa_enabled.set(False)
            self.app.toggle_qa()
            self.assertFalse(cancellation.is_set())
            self.app.open_qa()
            self.assertEqual(self.app.qa.generation, generation)
        finally:
            release.set()
        self.finish()
        self.assertEqual(self.app.qa_history, [('问题', '完成')])

    def test_clipboard_sends_while_editing_and_preserves_draft(self):
        self.enable()
        self.app.qa_question, self.app.qa_answer = '原问题', '原回答'
        self.app.render_qa()
        self.root.update()
        x, y, _, _ = self.app.qa_text.bbox('1.0')
        self.app.edit_question(SimpleNamespace(x=x+2, y=y+2))
        editor = self.app.question_editor
        editor.insert('end', '未发送草稿')
        self.app.qa.stream_runner = None
        self.app.qa.runner = lambda prompt, cancel: '新回答'
        self.enqueue(ClipboardItem('复制的问题'))
        self.app.send_pending_clipboard()
        self.finish()
        self.assertIs(self.app.question_editor, editor)
        self.assertEqual(editor.get('1.0', 'end-1c'), '原问题未发送草稿')
        self.assertEqual(self.app.qa_history, [('复制的问题', '新回答')])
        self.app.cancel_question_edit()

    def test_unified_switch_cancels_clipboard_and_discards_queued_voice(self):
        self.enable()
        self.app.qa.feed('语音问题？')
        self.app.qa.active = True
        self.app.qa.source = 'clipboard'
        self.app.clipboard_request_generation = self.app.qa.generation
        cancellation = self.app.qa.cancel
        self.app.auto_button.invoke()
        self.assertTrue(cancellation.is_set())
        self.assertFalse(self.app.auto_qa.get())
        self.assertFalse(self.app.qa.pending)
        self.assertFalse(self.app.clipboard_pending)
        self.app.clipboard_watcher.stop.assert_called()

    def test_icon_switch_enables_both_sources_and_hover_description(self):
        self.root.update()
        self.app.open_qa()
        self.app.auto_button.invoke()
        self.root.update()
        self.assertTrue(self.app.auto_qa.get())
        self.app.clipboard_watcher.start.assert_called_once()
        self.app.qa.feed('转录问题是什么？')
        self.assertTrue(self.app.qa.pending)
        self.enqueue(ClipboardItem('复制内容'))
        self.assertEqual(len(self.app.clipboard_pending), 1)
        self.app.auto_button.show_tooltip()
        labels = self.app.auto_button.tooltip_window.winfo_children()
        self.assertEqual(labels[0].cget('text'), '自动读取转录问题与剪贴板内容进行答疑')
        self.app.auto_button.hide_tooltip()
        self.app.auto_button.invoke()
        self.assertFalse(self.app.auto_qa.get())
        self.assertFalse(self.app.qa.pending)
        self.assertFalse(self.app.clipboard_pending)

    def test_deepseek_image_queue_limit_and_error_keeps_listening(self):
        self.enable()
        self.app.qa_settings['provider'] = 'deepseek'
        self.app.qa_settings['profiles']['deepseek'] = {'model': 'deepseek-v4-pro'}
        self.enqueue(ClipboardItem('图片', b'image'))
        self.assertEqual(self.app.clipboard_pending[0].image, b'image')
        self.app.clipboard_pending.clear()
        self.app.qa_settings['profiles']['deepseek']['model'] = 'deepseek-flash'
        self.enqueue(ClipboardItem('图片', b'image'))
        self.assertEqual(self.app.clipboard_pending[0].image, b'image')
        self.app.clipboard_pending.clear()
        for index in range(9):
            self.enqueue(ClipboardItem(str(index)))
        self.assertEqual(len(self.app.clipboard_pending), 8)
        self.assertIn('跳过', self.app.clipboard_status.get())
        self.app.clipboard_request_generation = self.app.qa.generation
        self.app.handle_qa((self.app.qa.generation, 'error', '服务错误'))
        self.assertTrue(self.app.auto_qa.get())
        self.assertEqual(len(self.app.clipboard_pending), 8)
        self.assertEqual(self.app.qa_status.get(), '服务错误')
        self.app.handle_qa((self.app.qa.generation, 'done', None))
        self.app.send_pending_clipboard()
        self.assertEqual(len(self.app.clipboard_pending), 8)
        self.assertIn('检查模型连接', self.app.clipboard_status.get())


if __name__ == '__main__':
    unittest.main()
