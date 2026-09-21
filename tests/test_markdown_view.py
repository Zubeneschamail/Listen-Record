import tkinter as tk
from tkinter import font as tkfont
import unittest
from unittest.mock import patch

from markdown_view import MarkdownView


class MarkdownViewTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.geometry('620x500')
        self.text = tk.Text(self.root, wrap='word', padx=14, font=('Segoe UI', 10))
        self.text.pack(fill='both', expand=True)
        self.root.update()
        self.view = MarkdownView(self.text, 10)

    def tearDown(self):
        self.root.destroy()

    def visible(self):
        return self.text.get('1.0', 'end-1c')

    def test_heading_nested_emphasis_lists_quote_and_code(self):
        self.view.insert('# 标题\n\n**加粗 *且斜体*** ~~删除~~ `a * b`\n\n'
                         '> 引用\n\n- 第一项\n  - 子项\n- [x] 完成\n\n'
                         '```python\nprint("**原样代码**")\n```')
        text = self.visible()
        self.assertTrue(text.startswith('标题\n'))
        self.assertIn('加粗 且斜体 删除 a * b', text)
        self.assertIn('• 第一项\n• 子项', text)
        self.assertIn('☑ 完成', text)
        self.assertIn('print("**原样代码**")', text)
        self.assertNotIn('```', text)
        self.assertTrue(self.text.tag_ranges('md:heading1'))
        self.assertTrue(self.text.tag_ranges('md:emphasis:strong'))
        font = tkfont.Font(root=self.text, font=self.text.tag_cget('md:emphasis:strong', 'font'))
        self.assertEqual(font.actual('weight'), 'bold')
        self.assertEqual(font.actual('slant'), 'italic')
        self.assertTrue(self.text.tag_ranges('md:list1'))
        self.assertTrue(self.text.tag_ranges('md:quote'))

    def test_unclosed_fence_and_escaped_markers_preserve_content(self):
        self.view.insert('\\*不是强调\\*\n\n```python\na = 1\nb = "**text**"')
        self.assertIn('*不是强调*', self.visible())
        self.assertIn('b = "**text**"', self.visible())
        self.assertTrue(self.text.tag_ranges('md:codeblock'))
        self.assertFalse(self.text.tag_ranges('md:strong'))

    def test_links_need_click_and_dangerous_schemes_and_html_never_execute(self):
        with patch('markdown_view.webbrowser.open') as open_url:
            self.view.insert('[文档](https://example.com/docs) [危险](javascript:alert) '
                             '![远程图片](https://example.com/picture.png) <script>alert(1)</script>')
            open_url.assert_not_called()
            self.assertEqual(set(self.view.links.values()), {'https://example.com/docs', 'https://example.com/picture.png'})
            self.assertFalse(self.text.image_names())
            self.view.open_link('javascript:alert(1)')
            self.view.open_link('file:///C:/test.exe')
            open_url.assert_not_called()
            self.view.open_link('https://example.com/docs')
            open_url.assert_called_once_with('https://example.com/docs')

    def test_tables_keep_cells_in_wide_and_narrow_panes(self):
        source = '| 名称 | 状态 | 说明 |\n|---|---|---|\n| 转录 | 正常 | 本地识别 |\n| 答疑 | 待配置 | 填写密钥 |'
        for width in (620, 150):
            with self.subTest(width=width):
                self.root.geometry(f'{width}x500')
                self.root.update()
                self.text.delete('1.0', 'end')
                self.view.insert(source)
                for value in ('转录', '正常', '本地识别', '答疑', '待配置', '填写密钥'):
                    self.assertIn(value, self.visible())
                self.assertNotIn('---', self.visible())

    def test_font_and_theme_updates_keep_existing_tags(self):
        self.view.insert('**粗体** `代码`\n\n> 引用')
        self.view.configure(13, True)
        font = tkfont.Font(root=self.text, font=self.text.tag_cget('md:strong', 'font'))
        self.assertEqual(font.actual('size'), 13)
        self.assertEqual(font.actual('weight'), 'bold')
        self.assertEqual(self.text.tag_cget('md:code', 'background'), '#222C3B')
        self.assertEqual(self.text.tag_cget('md:quote', 'foreground'), '#AEBBCD')

    def test_stream_append_preserves_selection_and_does_not_reconfigure_fonts(self):
        self.text.mark_set('answer_start', '1.0')
        self.text.mark_gravity('answer_start', 'left')
        source = '**重点** 😀 已有内容'
        spans = self.view.insert(source)
        self.text.tag_add('sel', '1.0', '1.2')
        with patch.object(self.text, 'delete', wraps=self.text.delete) as delete, \
                patch.object(self.text, 'tag_configure', wraps=self.text.tag_configure) as configure:
            spans = self.view.update_tail(source+'，接着回答', 'answer_start', spans)
            delete.assert_not_called()
            configure.assert_not_called()
        self.assertEqual(self.visible(), '重点 😀 已有内容，接着回答')
        self.assertEqual(self.text.get('sel.first', 'sel.last'), '重点')

    def test_incremental_markdown_matches_fresh_render_when_markup_closes(self):
        self.text.mark_set('answer_start', '1.0')
        self.text.mark_gravity('answer_start', 'left')
        spans = []
        fresh = tk.Text(self.root)
        view = MarkdownView(fresh, 10)
        for source in ('开头 😀\n\n**逐步', '开头 😀\n\n**逐步加粗**',
                       '开头 😀\n\n**逐步加粗**\n\n```python\na = 1',
                       '开头 😀\n\n**逐步加粗**\n\n```python\na = 1\n```\n\n最后一段',
                       '更正后的内容'):
            spans = self.view.update_tail(source, 'answer_start', spans)
            fresh.delete('1.0', 'end')
            view.insert(source)
            self.assertEqual(self.visible(), fresh.get('1.0', 'end-1c'))
            for tag in ('md:strong', 'md:codeblock'):
                self.assertEqual(tuple(map(str, self.text.tag_ranges(tag))), tuple(map(str, fresh.tag_ranges(tag))))


if __name__ == '__main__':
    unittest.main()
