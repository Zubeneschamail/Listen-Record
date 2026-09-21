from decimal import Decimal
import queue
import threading
import unittest
from unittest.mock import patch

import httpx

from qa_balance import Balance, BalanceQuery, balance_display, query_balance


class BalanceTests(unittest.TestCase):
    def test_official_get_returns_balances_without_chat_payload(self):
        def handler(request):
            self.assertEqual(str(request.url), 'https://api.deepseek.com/user/balance')
            self.assertEqual(request.method, 'GET')
            self.assertEqual(request.headers['Authorization'], 'Bearer fake')
            self.assertEqual(request.content, b'')
            return httpx.Response(200, json={'is_available': True, 'balance_infos': [
                {'currency': 'CNY', 'total_balance': '25.50'},
                {'currency': 'USD', 'total_balance': '5.00'}]})
        result = query_balance('deepseek', {'api_key': 'fake'}, threading.Event(), httpx.MockTransport(handler))
        self.assertEqual(result.amounts, {'CNY': Decimal('25.50'), 'USD': Decimal('5')})

    def test_amounts_keep_currency_and_show_zero_or_negative_balances(self):
        balance = Balance(True, {'CNY': Decimal('25.50'), 'USD': Decimal('5')})
        self.assertEqual(balance_display(balance), ('¥25.50 · US$5.00', ''))
        self.assertEqual(balance_display(Balance(False, {'CNY': Decimal('0')})),
                         ('¥0.00', '账户当前不可调用'))
        self.assertEqual(balance_display(Balance(False, {'CNY': Decimal('-1')})),
                         ('¥-1.00', '账户当前不可调用'))
        self.assertNotIn('%', balance_display(balance)[0])

    def test_errors_and_redirects_do_not_expose_credentials(self):
        for code in (401, 403, 429, 500, 307):
            calls = []
            def handler(request):
                calls.append(request)
                return httpx.Response(code, text='fake-key-sensitive', headers={'Location': 'https://elsewhere.invalid'})
            with self.assertRaises(RuntimeError) as caught:
                query_balance('deepseek', {'api_key': 'fake-key-sensitive'}, threading.Event(), httpx.MockTransport(handler))
            self.assertNotIn('fake-key-sensitive', str(caught.exception))
            self.assertEqual(len(calls), 1)

    def test_bad_responses_and_network_failure(self):
        responses = [None, {}, {'is_available': True, 'balance_infos': []},
                     {'is_available': True, 'balance_infos': [{'currency': 'CNY', 'total_balance': 'NaN'}]},
                     {'is_available': True, 'balance_infos': [{'currency': 'CNY', 'total_balance': '1e10000'}]}]
        for response in responses:
            with self.subTest(response=response), self.assertRaisesRegex(RuntimeError, '格式无效'):
                query_balance('deepseek', {'api_key': 'fake'}, threading.Event(),
                              httpx.MockTransport(lambda r: httpx.Response(200, json=response)))
        def timeout(request):
            raise httpx.ReadTimeout('sensitive text')
        with self.assertRaisesRegex(RuntimeError, '超时'):
            query_balance('deepseek', {'api_key': 'fake'}, threading.Event(), httpx.MockTransport(timeout))

    def test_unsupported_missing_key_and_cancel_never_request(self):
        def handler(request):
            self.fail('Unexpected request')
        cancel = threading.Event()
        for provider, profile in [('compatible', {'api_key': 'fake'}), ('deepseek', {})]:
            with self.assertRaises(RuntimeError):
                query_balance(provider, profile, cancel, httpx.MockTransport(handler))
        cancel.set()
        with self.assertRaisesRegex(RuntimeError, '已取消'):
            query_balance('deepseek', {'api_key': 'fake'}, cancel, httpx.MockTransport(handler))

    def test_async_completion_and_cancellation(self):
        events = queue.Queue()
        worker = BalanceQuery(events)
        answer = Balance(True, {'CNY': Decimal(20)})
        with patch('qa_balance.query_balance', return_value=answer):
            worker.start('deepseek', {'api_key': 'fake'})
            kind, (revision, result, error) = events.get(timeout=2)
        self.assertEqual((kind, revision, result, error), ('qa_balance', worker.revision, answer, ''))
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        def delayed(*args):
            entered.set()
            release.wait(2)
            finished.set()
            return answer
        worker.close()
        with patch('qa_balance.query_balance', side_effect=delayed) as query:
            worker.start('deepseek', {})
            self.assertTrue(entered.wait(2))
            worker.start('deepseek', {})
            self.assertEqual(query.call_count, 1)
            worker.close()
            release.set()
            self.assertTrue(finished.wait(2))
        self.assertTrue(events.empty())
