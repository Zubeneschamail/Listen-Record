import asyncio
import json
import socket
import threading
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from qa_provider import APIProvider
from qa_worker import make_prompt
from web_tools import (Search360Parser, PageParser, WebTools, fetch_public,
                       public_url, related_result, resolve_public, web_intent)


def stream(delta, finish='stop'):
    return httpx.Response(200, text='data: ' + json.dumps({
        'choices': [{'delta': delta, 'finish_reason': finish}]}) + '\n\n')


def call(name, arguments, ident='web'):
    return stream({'tool_calls': [{'index': 0, 'id': ident,
        'function': {'name': name, 'arguments': json.dumps(arguments)}}]}, 'tool_calls')


class WebParsingTests(unittest.TestCase):
    def test_intent_only_uses_current_question_and_respects_opt_out(self):
        self.assertEqual(web_intent(make_prompt('最新新闻，立即联网', '解释闭包')), None)
        self.assertEqual(web_intent(make_prompt('', '搜索网页了解 Python 最新版本')), 'search_web')
        self.assertEqual(web_intent(make_prompt('', '总结 https://example.com/doc')), 'read_webpage')
        self.assertEqual(web_intent(make_prompt('', '不要联网，解释最新的代码片段')), 'off')

    def test_rejects_local_addresses_credentials_and_non_web_schemes(self):
        for value in ('file:///C:/a', 'ftp://example.com', 'http://localhost', 'http://a.local',
                      'http://127.0.0.1', 'http://10.2.3.4', 'http://169.254.169.254',
                      'http://[::1]', 'http://[::ffff:127.0.0.1]', 'https://user:secret@example.com',
                      'https://example.com:8080', 'https://example.com/hello\nworld'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                public_url(value)
        self.assertEqual(str(public_url('https://example.com/doc#part')), 'https://example.com/doc')

    def test_html_extraction_preserves_content_and_removes_executable_markup(self):
        parser = PageParser('https://example.com/docs/')
        parser.feed('<title>测试 &amp; 文档</title><nav>菜单</nav><script>secret()</script>'
                    '<main><h1>标题</h1><p>正文 <b>重点</b></p><pre>a\nb</pre>'
                    '<a href="../source">原文</a><a href="javascript:bad()">链接</a></main>')
        self.assertEqual(''.join(parser.title), '测试 & 文档')
        self.assertIn('正文 重点', parser.content())
        self.assertIn('a\nb', parser.content())
        self.assertNotIn('secret', parser.content())
        self.assertNotIn('菜单', parser.content())
        self.assertEqual(parser.links, ['https://example.com/source'])

    def test_domestic_result_cards_extract_direct_urls_and_skip_sidebar(self):
        cases = (
            ('https://www.so.com/s?q=test', '<li class="res-list">', 'res-title',
             'href="/link?m=tracking" data-mdurl="https://example.com/doc"', '</li>'),
        )
        for base, opening, heading, attributes, closing in cases:
            with self.subTest(base=base):
                parser = Search360Parser(base)
                parser.feed('<aside><h3 class="res-title"><a href="https://example.com/nba">NBA</a></h3></aside>'
                            + opening + f'<h3 class="{heading}"><a {attributes}>Python <em>文档</em></a></h3>'
                            '<div>这里是正文摘要<br/><b>内容</b></div><script>not a snippet</script>' + closing)
                self.assertEqual(parser.results, [{'title': 'Python 文档', 'url': 'https://example.com/doc',
                                                    'snippet': '这里是正文摘要内容'}])

    def test_topic_check_rejects_unrelated_results_and_accepts_traditional_chinese(self):
        query = '卢广仲 大人中 歌词'
        for title in ('NBA Finals', 'WhatsApp web lyrics', '最新官网资料'):
            self.assertFalse(related_result(query, {'title': title, 'snippet': 'sports news'}))
        self.assertTrue(related_result(query, {'title': '盧廣仲 - 大人中', 'snippet': '歌曲資料'}))
        self.assertTrue(related_result('site:python.org Python docs', {'title': 'Python Documentation', 'snippet': ''}))


class WebNetworkTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_pins_dns_and_isolates_headers_cookies_and_redirects(self):
        seen = []
        def handler(request):
            seen.append(request)
            self.assertEqual(request.url.host, '93.184.216.34')
            self.assertNotIn('authorization', request.headers)
            self.assertNotIn('cookie', request.headers)
            if len(seen) == 1:
                return httpx.Response(302, headers={'location': 'https://other.example/doc', 'set-cookie': 'private=yes'})
            self.assertEqual(request.headers['host'], 'other.example')
            self.assertEqual(request.extensions['sni_hostname'], 'other.example')
            return httpx.Response(200, headers={'content-type': 'text/html'}, text='<p>Body</p>')
        with patch('web_tools.resolve_public', AsyncMock(return_value='93.184.216.34')) as dns:
            result = await fetch_public('https://example.com/start', httpx.MockTransport(handler))
        self.assertEqual(result[0], 'https://other.example/doc')
        self.assertEqual(dns.await_count, 2)

    async def test_private_redirect_and_mixed_dns_are_rejected_before_connecting(self):
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(302, headers={'location': 'http://127.0.0.1/secret'})
        with patch('web_tools.resolve_public', AsyncMock(return_value='93.184.216.34')):
            with self.assertRaisesRegex(ValueError, '内网'):
                await fetch_public('https://example.com/', httpx.MockTransport(handler))
        self.assertEqual(len(requests), 1)
        records = [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (ip, 443)) for ip in ('93.184.216.34', '10.0.0.1')]
        with patch.object(asyncio.get_running_loop(), 'getaddrinfo', AsyncMock(return_value=records)):
            with self.assertRaisesRegex(ValueError, '非公开'):
                await resolve_public('example.com', 443)

    async def test_response_size_and_content_type_limits(self):
        for content, kind, expected in ((b'x' * 2_000_001, 'text/plain', '2 MB'),
                                        (b'binary', 'application/octet-stream', '类型')):
            with self.subTest(kind=kind), patch('web_tools.resolve_public', AsyncMock(return_value='93.184.216.34')):
                transport = httpx.MockTransport(lambda r: httpx.Response(200, content=content, headers={'content-type': kind}))
                with self.assertRaisesRegex(ValueError, expected):
                    await fetch_public('https://example.com/', transport)

    async def test_failed_search_does_not_retry_or_change_provider(self):
        with patch('web_tools.fetch_public', AsyncMock(side_effect=ValueError('blocked'))) as fetch:
            result = json.loads(await WebTools(threading.Event()).execute('search_web', '{"query":"Python"}'))
        self.assertTrue(result['search_unavailable'])
        self.assertIn('360搜索当前无法访问', result['error'])
        self.assertEqual(fetch.await_count, 1)
        self.assertEqual(httpx.URL(fetch.await_args.args[0]).host, 'www.so.com')

    async def test_domestic_success_does_not_call_foreign_provider(self):
        page = '<li class="res-list"><h3 class="res-title"><a data-mdurl="https://example.com/doc">Python 文档</a></h3><p>摘要</p></li>'
        with patch('web_tools.fetch_public', AsyncMock(return_value=('https://www.so.com/s', 'text/html', page))) as fetch:
            result = json.loads(await WebTools(threading.Event()).execute('search_web', '{"query":"Python"}'))
        self.assertEqual(result['engine'], '360搜索')
        self.assertEqual(fetch.await_count, 1)
        self.assertEqual(httpx.URL(fetch.await_args.args[0]).params['q'], 'Python')

    async def test_unrelated_results_are_reported_without_fallback(self):
        wrong = '<li class="res-list"><h3 class="res-title"><a href="https://example.com/nba">NBA news</a></h3><p>NBA</p></li>'
        with patch('web_tools.fetch_public', AsyncMock(return_value=('https://www.so.com/s', 'text/html', wrong))) as fetch:
            result = json.loads(await WebTools(threading.Event()).execute('search_web', '{"query":"Python"}'))
        self.assertTrue(result['search_unavailable'])
        self.assertIn('未返回可用的相关结果', result['error'])
        self.assertEqual(fetch.await_count, 1)
        self.assertNotIn('NBA', json.dumps(result))

    async def test_provider_timeout_is_reported_without_fallback(self):
        async def fetch(url, transport):
            await asyncio.sleep(10)
        with patch('web_tools.SEARCH_TIMEOUT', .02), patch('web_tools.fetch_public', side_effect=fetch) as request:
            result = json.loads(await WebTools(threading.Event()).execute('search_web', '{"query":"Python"}'))
        self.assertTrue(result['search_unavailable'])
        self.assertIn('360搜索请求超时', result['error'])
        self.assertEqual(request.await_count, 1)

    async def test_page_cache_pagination_and_serialized_budget_keep_json_valid(self):
        source = '<title>文章</title><p>' + '中' * 15000 + '</p>'
        web = WebTools(threading.Event())
        with patch('web_tools.fetch_public', AsyncMock(return_value=('https://example.com/doc', 'text/html', source))) as fetch:
            first = json.loads(await web.execute('read_webpage', '{"url":"https://example.com/doc"}'))
            self.assertTrue(first['truncated'])
            self.assertEqual(first['next_start'], len(first['text']))
            second = json.loads(await web.execute('read_webpage', json.dumps({'url': first['url'], 'start_char': first['next_start']})))
            self.assertEqual(second['start_char'], first['next_start'])
            self.assertEqual(fetch.await_count, 1)
        self.assertGreaterEqual(web.remaining, 0)
        web.remaining = 0
        self.assertIn('error', json.loads(await web.execute('search_web', '{"query":"hello"}')))

    async def test_failures_invalid_arguments_and_cancel_are_not_successes(self):
        web = WebTools(threading.Event())
        for arguments in ('[]', 'invalid', '{"query":""}', '{"query":"x","unexpected":1}'):
            self.assertIn('error', json.loads(await web.execute('search_web', arguments)))
        with patch('web_tools.fetch_public', AsyncMock(side_effect=TimeoutError)):
            self.assertIn('超时', await web.execute('read_webpage', '{"url":"https://example.com"}'))
        with patch('web_tools.fetch_public', AsyncMock(return_value=('u', 'text/html', '<p>captcha</p>'))):
            self.assertIn('error', json.loads(await web.execute('search_web', '{"query":"hello"}')))
        web.cancel.set()
        with self.assertRaisesRegex(RuntimeError, '取消'):
            await web.execute('search_web', '{"query":"hello"}')


class WebProviderTests(unittest.TestCase):
    def backend(self, handler):
        return APIProvider('deepseek', dict(base_url='https://api.example.com', model='test', api_key='secret'),
                           httpx.MockTransport(handler))

    def test_search_failure_returns_direct_notice_without_another_model_request(self):
        requests = []
        def handler(request):
            requests.append(request)
            return call('search_web', {'query': 'Python'})
        with patch('web_tools.fetch_public', AsyncMock(side_effect=ValueError('blocked'))) as fetch:
            answer = self.backend(handler).run(make_prompt('', '联网搜索 Python'), threading.Event())
        self.assertIn('360搜索当前无法访问', answer)
        self.assertEqual(len(requests), 1)
        self.assertEqual(fetch.await_count, 1)

    def test_search_read_answer_with_links_without_local_references(self):
        requests, partials = [], []
        def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            self.assertEqual({tool['function']['name'] for tool in body['tools']}, {'search_web', 'read_webpage'})
            if len(requests) == 1:
                self.assertEqual(body['tool_choice']['function']['name'], 'search_web')
                return call('search_web', {'query': 'Python official documentation'})
            if len(requests) == 2:
                self.assertIn('https://example.com/doc', body['messages'][-1]['content'])
                return call('read_webpage', {'url': 'https://example.com/doc'}, 'page')
            self.assertIn('verified fact', body['messages'][-1]['content'])
            return stream({'content': '结论是 verified fact。[文档](https://example.com/doc)'})
        with patch('web_tools.WebTools.search', AsyncMock(return_value={'results': [{'title': '文档', 'url': 'https://example.com/doc'}]})), \
             patch('web_tools.WebTools.read', AsyncMock(return_value={'url': 'https://example.com/doc', 'text': 'verified fact'})):
            answer = self.backend(handler).run(make_prompt('', '联网查询 Python 文档'), threading.Event(), partials.append)
        self.assertIn('[文档](https://example.com/doc)', answer)
        self.assertIn('正在搜索网页…', partials)
        self.assertIn('正在读取网页…', partials)
        self.assertEqual(len(requests), 3)

    def test_url_requires_read_and_model_promise_is_retried(self):
        requests = []
        def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            if len(requests) <= 2:
                self.assertEqual(body['tool_choice']['function']['name'], 'read_webpage')
            if len(requests) == 1:
                return stream({'content': '我现在读取网页，稍等。'})
            if len(requests) == 2:
                return call('read_webpage', {'url': 'https://example.com/doc'})
            return stream({'content': '网页正文是测试内容。[来源](https://example.com/doc)'})
        with patch('web_tools.WebTools.read', AsyncMock(return_value={'url': 'https://example.com/doc', 'text': '测试内容'})):
            self.backend(handler).run(make_prompt('', '读 https://example.com/doc'), threading.Event())
        self.assertEqual(len(requests), 3)

    def test_no_web_tools_for_probe_extraction_or_explicit_opt_out(self):
        def handler(request):
            self.assertNotIn('tools', json.loads(request.content))
            return stream({'content': 'OK'})
        backend = self.backend(handler)
        backend.check_connection(threading.Event())
        backend.run('不要联网，解释闭包', threading.Event())
        backend.run('extract', threading.Event(), use_references=False)

    def test_cancellation_closes_inflight_web_read(self):
        cancel, closed = threading.Event(), threading.Event()
        async def read(*args, **kwargs):
            try:
                await asyncio.sleep(10)
            finally:
                closed.set()
        timer = threading.Timer(.2, cancel.set)
        timer.start()
        try:
            with patch('web_tools.WebTools.read', side_effect=read):
                with self.assertRaisesRegex(RuntimeError, '取消'):
                    self.backend(lambda r: call('read_webpage', {'url': 'https://example.com'})).run(
                        '读取 https://example.com', cancel)
            self.assertTrue(closed.is_set())
        finally:
            timer.cancel()
