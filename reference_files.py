"""User-selected, read-only reference files exposed through bounded tools."""
import fnmatch
import json
import os
from pathlib import Path, PurePosixPath
import re
import time
import zipfile
from xml.etree import ElementTree

from app_paths import DATA

SETTINGS = DATA / 'reference-files.json'
TEXT_EXTENSIONS = set('.txt .md .rst .csv .tsv .json .jsonl .yaml .yml .toml .xml .html .css .js .jsx .ts .tsx .py .cs .c .h .cpp .hpp .java .go .rs .sql .sh .ps1 .bat .ini .cfg .log'.split())
SUPPORTED = TEXT_EXTENSIONS | {'.pdf', '.docx'}
IGNORED = {'.git', '.svn', '.hg', 'node_modules', '.venv', 'venv', '__pycache__', 'dist', 'build', '.idea', '.vscode', '.ssh', '.aws'}
PRIVATE = {'.env', 'credentials', 'credentials.json', 'secrets.json', 'qa-settings.json', 'id_rsa', 'id_ed25519'}


def excluded(path):
    return any(part.lower() in IGNORED for part in path.parts) or path.name.lower() in PRIVATE or path.name.lower().startswith('.env.') or path.suffix.lower() in {'.pem', '.key', '.pfx', '.p12'}


def load_settings(path=SETTINGS):
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        paths = data.get('paths', [])
        if not isinstance(paths, list):
            return {'enabled': False, 'paths': []}
        return {'enabled': data.get('enabled') is True,
                'paths': list(dict.fromkeys(p for p in paths if isinstance(p, str)))[:32]}
    except (OSError, ValueError, AttributeError):
        return {'enabled': False, 'paths': []}


def save_settings(settings, path=SETTINGS):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def validate_source(value):
    path = Path(value).resolve(strict=True)
    if excluded(path):
        raise ValueError('此位置属于密钥、凭据或默认忽略目录，请选择资料文件。')
    if not path.is_dir() and (not path.is_file() or path.suffix.lower() not in SUPPORTED):
        raise ValueError('支持文本、代码、PDF 和 DOCX；暂不支持此文件格式。')
    return str(path)


def schema(name, description, properties, required):
    return {'type': 'function', 'function': {'name': name, 'description': description,
            'parameters': {'type': 'object', 'properties': properties, 'required': required,
                           'additionalProperties': False}}}


TOOLS = [
    schema('list_reference_files', '列出用户添加的资料，支持按文件名筛选。返回的路径可用于读取或搜索。',
           {'pattern': {'type': 'string', 'description': '文件名通配符，如 *.md；默认 *'},
            'offset': {'type': 'integer', 'minimum': 0, 'description': '分页偏移，默认 0'}}, []),
    schema('search_reference_files', '在选定资料中搜索一个关键词或短语，返回文件路径、行号和原文。多关键词请分别调用。',
           {'query': {'type': 'string'}, 'path': {'type': 'string', 'description': '可选，限定资料文件或目录，如 r1/docs'}}, ['query']),
    schema('read_reference_file', '分段读取资料。文本返回行号，PDF 还标注页码；用这些信息引用来源。',
           {'path': {'type': 'string'}, 'start_line': {'type': 'integer', 'minimum': 1},
            'limit': {'type': 'integer', 'minimum': 1, 'maximum': 120}}, ['path']),
]


class ReferenceTools:
    def __init__(self, settings, cancel):
        self.cancel = cancel
        self.roots = {}
        if settings.get('enabled'):
            for i, value in enumerate(settings.get('paths', [])[:32]):
                try:
                    path = Path(validate_source(value))
                    self.roots[f'r{i+1}'] = path
                except (OSError, ValueError):
                    continue
        self.remaining = 32000  # Serialized tool-result bytes across the request.
        self.cache = {}
        self.deadline = 0
        self.truncated = False

    def description(self):
        return [{'root': key, 'name': path.name, 'kind': 'folder' if path.is_dir() else 'file'}
                for key, path in self.roots.items()]

    def check(self):
        if self.cancel.is_set():
            raise RuntimeError('已取消')
        if time.monotonic() > self.deadline:
            raise ValueError('本次检索达到时间上限，请指定更小的目录或文件。')

    def resolve(self, virtual):
        if not isinstance(virtual, str) or len(virtual) > 2048 or '\\' in virtual or ':' in virtual:
            raise ValueError('请使用工具返回的相对资料路径。')
        parts = PurePosixPath(virtual).parts
        if not parts or parts[0] not in self.roots or any(p in ('.', '..') for p in parts):
            raise ValueError('路径不在已添加的参考资料内。')
        root = self.roots[parts[0]]
        if root.is_file():
            if tuple(parts[1:]) != (root.name,):
                raise ValueError('只能读取明确添加的文件。')
            if root.is_symlink() or root.resolve(strict=True) != root:
                raise ValueError('资料文件位置已改变，请重新添加。')
            path = root
        else:
            path = root.joinpath(*parts[1:]).resolve(strict=True)
            if not path.is_relative_to(root):
                raise ValueError('不允许访问参考目录之外的内容。')
        if excluded(path):
            raise ValueError('该文件或目录已被排除。')
        return path

    def files(self, scope=''):
        count = 0
        for key, root in self.roots.items():
            initial = [(root, f'{key}/{root.name}')] if root.is_file() else [(root, key)]
            stack = initial
            while stack:
                self.check()
                path, virtual = stack.pop()
                count += 1
                if count > 10000:
                    self.truncated = True
                    return
                if scope and not (virtual == scope or virtual.startswith(scope+'/') or scope.startswith(virtual+'/')):
                    continue
                try:
                    # No symlink/junction traversal; resolved paths must remain scoped.
                    if path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction()) or excluded(path):
                        continue
                    resolved = path.resolve(strict=True)
                    if resolved != root and not resolved.is_relative_to(root):
                        continue
                    if path.is_dir():
                        with os.scandir(path) as entries:
                            children = []
                            for entry in entries:
                                self.check()
                                if len(children) + count > 10000:
                                    self.truncated = True
                                    break
                                children.append((Path(entry.path), virtual+'/'+entry.name))
                            stack.extend(sorted(children, key=lambda pair: pair[1], reverse=True))
                    elif path.suffix.lower() in SUPPORTED:
                        yield virtual, resolved
                except OSError:
                    continue

    def lines(self, path):
        self.check()
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        if key in self.cache:
            return self.cache[key]
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED:
            raise ValueError('不支持读取此文件格式。')
        if stat.st_size > (20_000_000 if suffix in {'.pdf', '.docx'} else 2_000_000):
            raise ValueError('文件过大，请拆分资料或选择较小文件。')
        if suffix == '.docx':
            with zipfile.ZipFile(path) as archive:
                if archive.getinfo('word/document.xml').file_size > 8_000_000:
                    raise ValueError('DOCX 正文过大，请拆分后添加。')
                tree = ElementTree.fromstring(archive.read('word/document.xml'))
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            text = '\n'.join(''.join(p.itertext()) for p in tree.findall('.//w:p', ns))
        elif suffix == '.pdf':
            from pypdf import PdfReader
            reader = PdfReader(path)
            if reader.is_encrypted:
                raise ValueError('PDF 已加密，请先提供可读取版本。')
            if len(reader.pages) > 100:
                raise ValueError('PDF 超过 100 页，请按章节拆分后添加。')
            pages = []
            for i, page in enumerate(reader.pages):
                self.check()
                pages.append(f'[第 {i+1} 页]\n'+(page.extract_text() or ''))
                if sum(len(p) for p in pages) > 200000:
                    raise ValueError('PDF 文字过多，请按章节拆分后添加。')
            text = '\n'.join(pages)
            if not any(re.search(r'\w', p.split('\n', 1)[-1]) for p in pages):
                raise ValueError('PDF 没有可提取文字；扫描件需要先做 OCR。')
        else:
            raw = path.read_bytes()
            if raw.startswith((b'\xff\xfe', b'\xfe\xff')):
                text = raw.decode('utf-16')
            else:
                if b'\0' in raw:
                    raise ValueError('此文件是二进制文件，无法作为文本读取。')
                try:
                    text = raw.decode('utf-8-sig')
                except UnicodeDecodeError:
                    text = raw.decode('gb18030')
        self.check()
        if len(text) > 500000:
            raise ValueError('文件文字过多，请拆分后添加。')
        lines = text.splitlines()
        if len(self.cache) >= 8:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = lines
        return lines

    def dispatch(self, name, args):
        if name == 'list_reference_files':
            pattern = args.get('pattern', '*')
            offset = args.get('offset', 0)
            if not isinstance(pattern, str) or len(pattern) > 200 or type(offset) is not int or not 0 <= offset <= 10000:
                raise ValueError('文件筛选参数无效。')
            files = [v for v, _ in self.files() if fnmatch.fnmatch(v.lower(), pattern.lower()) or fnmatch.fnmatch(PurePosixPath(v).name.lower(), pattern.lower())]
            return {'files': files[offset:offset+60], 'next_offset': offset+60 if len(files)>offset+60 else None, 'scan_limited': self.truncated}
        if name == 'read_reference_file':
            path = self.resolve(args.get('path'))
            start, limit = args.get('start_line', 1), args.get('limit', 80)
            if type(start) is not int or start < 1 or type(limit) is not int or not 1 <= limit <= 120:
                raise ValueError('行号或读取数量无效。')
            lines = self.lines(path)
            output = []
            for index in range(start-1, min(start-1+limit, len(lines))):
                output.append(f'{index+1}: {lines[index][:1200]}')
                if sum(len(line) for line in output) > 3500:
                    break
            return {'path': args['path'], 'lines': output, 'total_lines': len(lines),
                    'next_line': start+len(output) if start+len(output)<=len(lines) else None}
        if name == 'search_reference_files':
            query, scope = args.get('query'), args.get('path', '')
            if not isinstance(query, str) or not query.strip() or len(query) > 200 or not isinstance(scope, str):
                raise ValueError('请提供 1–200 字的检索关键词。')
            if scope:
                self.resolve(scope)
            results, skipped, scanned = [], 0, 0
            for virtual, path in self.files(scope):
                self.check()
                scanned += 1
                if scanned > 200:
                    self.truncated = True
                    break
                try:
                    lines = self.lines(path)
                except (OSError, ValueError, zipfile.BadZipFile, ElementTree.ParseError):
                    skipped += 1
                    continue
                for i, line in enumerate(lines):
                    self.check()
                    offset = line.casefold().find(query.casefold())
                    if offset >= 0:
                        results.append({'path': virtual, 'line': i+1, 'text': line[max(0, offset-100):offset+220]})
                        if len(results) >= 20:
                            return {'matches': results, 'limited': True, 'skipped_files': skipped}
            return {'matches': results, 'limited': self.truncated, 'skipped_files': skipped}
        raise ValueError('此工具不在允许的只读工具列表中。')

    def execute(self, name, arguments):
        self.deadline = time.monotonic()+5
        if self.remaining < 1000:
            return json.dumps({'error': '本次资料读取额度已用完，请根据已有资料回答。'}, ensure_ascii=False)
        try:
            self.check()
            args = json.loads(arguments)
            if not isinstance(args, dict):
                raise ValueError('工具参数必须为对象。')
            result = self.dispatch(name, args)
        except RuntimeError:
            raise
        except Exception:
            # Do not disclose absolute paths or internal exception contents to the API.
            result = {'error': '无法读取或检索该资料：请检查路径、格式、大小或权限；可换一个文件或缩小范围。'}
        encoded = json.dumps(result, ensure_ascii=False)
        # Keep JSON valid when limiting unusually large listings/results.
        if len(encoded.encode('utf-8')) > min(10000, self.remaining):
            original = encoded.encode('utf-8')
            length = min(8500, self.remaining-300)
            while True:
                preview = original[:length].decode('utf-8', errors='ignore')
                encoded = json.dumps({'truncated': True, 'excerpt': preview}, ensure_ascii=False)
                if len(encoded.encode('utf-8')) <= min(10000, self.remaining):
                    break
                length = length * 3 // 4
        self.remaining -= len(encoded.encode('utf-8'))
        return encoded
