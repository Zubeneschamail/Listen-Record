import gc
import tkinter as tk
import unittest
from unittest.mock import patch

from app import App


class QuestionFocusTests(unittest.TestCase):
    def setUp(self):
        gc.collect()
        for name,options in (
                ('app.GlobalHotkey',{}),('app.App.start_tray',{}),('app.ClipboardWatcher',{}),
                ('qa_connection.QAConnection.check',{}),('app.save_desktop',{}),
                ('app.preferences',{'return_value':{}}),
                ('app.load_settings',{'return_value':{'provider':'compatible','profiles':{}}})):
            item=patch(name,**options);item.start();self.addCleanup(item.stop)
        self.root=tk.Tk()
        self.app=App(self.root)
        self.app.open_qa()
        self.app.qa_display_history=[('第一个问题','第一个回答')]
        self.app.qa_question,self.app.qa_answer='第二个问题','第二个回答'
        self.app.render_qa()
        self.root.geometry('900x600')
        self.root.update()
        self.root.focus_force()
        self.app.qa_text.focus_set()
        self.root.update()

    def tearDown(self):
        self.app.qa.set_enabled(False)
        self.app.qa.context_provider=self.app.qa.voice_context_provider=None
        self.app.qa_connection.close();self.app.gpu_check.close();self.app.hotkey.close()
        for timer in self.root.tk.call('after','info'):self.root.after_cancel(timer)
        self.root.destroy()
        self.app=self.root=None
        gc.collect()

    def click_first_question(self):
        text=self.app.qa_text
        start=text.tag_ranges('question_turn:0')[0]
        text.see(start);self.root.update()
        x,y,_,_=text.bbox(start)
        text.event_generate('<Motion>',x=x+2,y=y+2)
        text.event_generate('<ButtonPress-1>',x=x+2,y=y+2)
        text.event_generate('<ButtonRelease-1>',x=x+2,y=y+2)
        return self.app.question_editor

    def test_one_real_click_routes_select_all_and_typing_to_the_question(self):
        editor=self.click_first_question()
        self.root.update()
        self.assertIs(self.root.focus_get(),editor)
        editor.event_generate('<Control-a>');self.root.update()
        self.assertEqual(editor.get('sel.first','sel.last').rstrip('\n'),'第一个问题')
        self.assertFalse(self.app.qa_text.tag_ranges('sel'))
        editor.tk.call('tk::TextInsert',editor,'修改第一题')
        self.assertEqual(editor.get('1.0','end-1c'),'修改第一题')
        self.assertEqual(self.app.qa_display_history,[('第一个问题','第一个回答')])

    def test_cancel_before_idle_does_not_focus_a_destroyed_editor(self):
        self.click_first_question()
        self.app.cancel_question_edit()
        self.app.composer.input.focus_set()
        self.root.update()
        self.assertIs(self.root.focus_get(),self.app.composer.input)

    def test_delayed_focus_respects_a_new_input_target(self):
        editor=self.click_first_question()
        self.app.composer.input.focus_set()
        self.root.update()
        self.assertIs(self.root.focus_get(),self.app.composer.input)
        self.assertEqual(editor.get('1.0','end-1c'),'第一个问题')
