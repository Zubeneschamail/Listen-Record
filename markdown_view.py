"""Markdown rendered as selectable native Tk text, with no HTML or remote fetches."""
from functools import lru_cache
import hashlib
import tkinter.font as tkfont
import unicodedata
from urllib.parse import urlsplit
import webbrowser

import mistune
from mistune.plugins.formatting import strikethrough
from mistune.plugins.table import table
from mistune.plugins.task_lists import task_lists
from mistune.plugins.url import url
import typography
from theme import color


_parse = mistune.create_markdown(renderer='ast', plugins=[strikethrough, table, task_lists, url])


@lru_cache(maxsize=96)
def parse(source):
    return _parse(source)


def safe_link(url):
    try:
        value = urlsplit(url)
        return value.scheme.lower() in ('http', 'https') and bool(value.netloc) and not any(ord(c) < 33 for c in url)
    except ValueError:
        return False


class MarkdownView:
    def __init__(self, widget, size=9, dark=False):
        self.widget = widget
        self.size, self.dark = size, dark
        self.styles, self.fonts, self.links = {}, {}, {}
        self.style_settings = {}

    def configure(self, size, dark):
        self.size, self.dark = size, dark
        for tag, styles in self.styles.items():
            self.style_tag(styles)

    def style_tag(self, styles):
        styles = tuple(sorted(set(styles)))
        if not styles:
            return None
        tag = 'md:' + ':'.join(styles)
        if self.style_settings.get(tag) == (self.size, self.dark):
            return tag
        self.styles[tag] = styles
        heading = next((int(s[-1]) for s in styles if s.startswith('heading')), 0)
        mono = any(s in styles for s in ('code', 'codeblock', 'table'))
        if tag not in self.fonts:
            self.fonts[tag] = tkfont.Font(root=self.widget)
        font = self.fonts[tag]
        font.configure(family='Consolas' if mono else typography.UI_FAMILY,
                       size=3 if 'gap' in styles else self.size + (max(1, 5-heading) if heading else 0),
                       weight='bold' if heading or 'strong' in styles else 'normal',
                       slant='italic' if 'emphasis' in styles else 'roman',
                       overstrike='strikethrough' in styles)
        options = dict(font=font)
        if 'gap' in styles:
            options.update(spacing1=0, spacing2=0, spacing3=0)
        if heading:
            options.update(spacing1=8, spacing3=5)
        if 'code' in styles or 'codeblock' in styles:
            options.update(background=color('#f1f3f7', self.dark))
        if 'codeblock' in styles or 'table' in styles:
            options.update(spacing1=0, spacing2=1, spacing3=0, lmargin1=6, lmargin2=6, rmargin=6)
        if 'quote' in styles:
            options.update(foreground=color('#737b8c', self.dark), lmargin1=12, lmargin2=12)
        depth = next((int(s[4:]) for s in styles if s.startswith('list')), None)
        if depth is not None:
            options.update(lmargin1=depth*14, lmargin2=depth*14+16, spacing1=1, spacing3=2)
        if 'link' in styles:
            options.update(foreground='#007ACC', underline=True)
        if 'rule' in styles:
            options.update(foreground=color('#e3e9f0', self.dark))
        self.widget.tag_configure(tag, **options)
        self.style_settings[tag] = (self.size, self.dark)
        return tag

    def link_tag(self, target):
        if not safe_link(target):
            return None
        tag = 'mdlink:' + hashlib.sha256(target.encode()).hexdigest()[:20]
        if tag not in self.links:
            self.links[tag] = target
            self.widget.tag_bind(tag, '<Button-1>', lambda event, value=target: self.open_link(value))
            self.widget.tag_bind(tag, '<Enter>', lambda event: self.widget.configure(cursor='hand2'))
            self.widget.tag_bind(tag, '<Leave>', lambda event: self.widget.configure(cursor='xterm'))
        return tag

    def open_link(self, target):
        if safe_link(target):
            webbrowser.open(target)
        return 'break'

    def insert(self, source, base_tags=()):
        spans = self.spans(source, base_tags)
        self.append_spans(spans)
        return spans

    def append_spans(self, spans):
        for text, tags in spans:
            self.widget.insert('end', text, tags)
        self.widget.tag_raise('sel')

    def update_tail(self, source, start, previous):
        """Retain the unchanged styled prefix, including selection and viewport."""
        spans = self.spans(source)
        old_i = new_i = old_offset = new_offset = 0
        common = []
        while old_i < len(previous) and new_i < len(spans):
            old_text, old_tags = previous[old_i]
            new_text, new_tags = spans[new_i]
            if old_tags != new_tags:
                break
            old, new = old_text[old_offset:], new_text[new_offset:]
            length = min(len(old), len(new))
            matched = length
            if old[:length] != new[:length]:
                matched = next(i for i in range(length) if old[i] != new[i])
            common.append(new[:matched])
            old_offset += matched
            new_offset += matched
            if matched < length:
                break
            if old_offset == len(old_text):
                old_i, old_offset = old_i+1, 0
            if new_offset == len(new_text):
                new_i, new_offset = new_i+1, 0
        # Text's +Nc counts Unicode characters; Tcl string length may count
        # surrogate pairs twice and would cut past the boundary after an emoji.
        count = len(''.join(common))
        boundary = self.widget.index(f'{start}+{count}c')
        if self.widget.compare(boundary, '<', 'end-1c'):
            self.widget.delete(boundary, 'end-1c')
        tail = spans[new_i:]
        if tail and new_offset:
            tail = [(tail[0][0][new_offset:], tail[0][1])] + tail[1:]
        self.append_spans(tail)
        return spans

    def spans(self, source, base_tags=()):
        spans = []
        def emit(text, styles=(), link=None):
            if text:
                if spans and spans[-1][1:] == (styles, link):
                    spans[-1] = (spans[-1][0]+text, styles, link)
                else:
                    spans.append((text, styles, link))

        def render(tokens, styles=(), link=None):
            for token in tokens:
                kind = token['type']
                children, attrs = token.get('children', []), token.get('attrs', {})
                if kind in ('text', 'inline_html', 'block_html'):
                    emit(token.get('raw', ''), styles, link)
                elif kind == 'blank_line':
                    emit('\n', ('gap',))
                elif kind in ('softbreak', 'linebreak'):
                    emit('\n', styles)
                elif kind in ('strong', 'emphasis', 'strikethrough'):
                    render(children, styles+(kind,), link)
                elif kind == 'codespan':
                    emit(token['raw'], styles+('code',), link)
                elif kind in ('link', 'image'):
                    target = attrs.get('url', '')
                    if kind == 'image':
                        emit('图片：', styles)
                    render(children, styles+(('link',) if safe_link(target) else ()), target)
                elif kind in ('paragraph', 'block_text', 'heading'):
                    block_style = styles + ((f"heading{attrs['level']}",) if kind == 'heading' else ())
                    render(children, block_style, link)
                    emit('\n', block_style)
                elif kind == 'block_code':
                    raw = token.get('raw', '')
                    emit(raw + ('' if raw.endswith('\n') else '\n'), styles+('codeblock',))
                elif kind == 'block_quote':
                    render(children, styles+('quote',))
                elif kind == 'list':
                    depth = attrs.get('depth', 0)
                    list_style = tuple(s for s in styles if not s.startswith('list')) + (f'list{depth}',)
                    for number, item in enumerate(children, attrs.get('start', 1)):
                        marker = f'{number}. ' if attrs.get('ordered') else '• '
                        if item['type'] == 'task_list_item':
                            marker = '☑ ' if item.get('attrs', {}).get('checked') else '☐ '
                        emit(marker, list_style)
                        render(item.get('children', []), list_style)
                elif kind == 'thematic_break':
                    emit('────────────────\n', styles+('rule',))
                elif kind == 'table':
                    self.render_table(token, emit, styles)
                else:
                    render(children, styles, link)
        render(parse(source))
        # The caller owns inter-message spacing. Do not add a trailing empty row.
        while spans and spans[-1][0].endswith('\n'):
            text, styles, link = spans[-1]
            text = text.rstrip('\n')
            if text:
                spans[-1] = (text, styles, link)
                break
            spans.pop()
        prepared = []
        for text, styles, target in spans:
            tags = tuple(base_tags) + tuple(t for t in (self.style_tag(styles), self.link_tag(target) if target and 'question' not in base_tags else None) if t)
            prepared.append((text, tags))
        return prepared

    def render_table(self, token, emit, styles):
        def plain(node):
            return node.get('raw', '') or ''.join(plain(child) for child in node.get('children', []))
        head = token['children'][0]['children']
        rows = [head] + [row['children'] for row in token['children'][1]['children']]
        count = len(head)
        measure = tkfont.Font(root=self.widget, family='Consolas', size=self.size)
        available = max(40, self.widget.winfo_width()-int(self.widget.cget('padx'))*2-20)
        units = max(10, int(available / max(1, measure.measure('0'))))
        width = (units-(count-1)*3)//max(1, count)
        if width < 6:
            # Narrow panes retain all cell content as labelled rows.
            for row_index, row in enumerate(rows[1:]):
                if row_index:
                    emit('\n', styles)
                for header, cell in zip(head, row):
                    emit(plain(header)+': ', styles+('strong',))
                    emit(plain(cell)+'\n', styles)
            return
        def char_width(char):
            return 2 if unicodedata.east_asian_width(char) in ('W', 'F') else 1
        def wrap(text):
            lines, current, used = [], '', 0
            for char in text.replace('\t', '    '):
                size = char_width(char)
                if char == '\n' or used+size > width:
                    lines.append(current+' '*(width-used))
                    current, used = '', 0
                if char != '\n':
                    current += char
                    used += size
            lines.append(current+' '*(width-used))
            return lines
        for row_index, row in enumerate(rows):
            cells = [wrap(plain(cell)) for cell in row]
            tags = styles+('table',)+(('strong',) if row_index == 0 else ())
            for line in range(max(map(len, cells))):
                emit(' │ '.join(cell[line] if line < len(cell) else ' '*width for cell in cells)+'\n', tags)
            if row_index == 0:
                emit('─┼─'.join('─'*width for _ in cells)+'\n', styles+('table', 'rule'))
