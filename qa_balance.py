"""Account balance lookup, separate from model requests and token usage."""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import threading

import httpx


@dataclass(frozen=True)
class Balance:
    available: bool
    amounts: dict


def query_balance(provider, profile, cancel, transport=None):
    if provider != 'deepseek':
        raise RuntimeError('当前兼容 API 暂不支持余额查询。')
    key = profile.get('api_key', '').strip()
    if not key:
        raise RuntimeError('请先在模型服务中保存 API Key。')
    if cancel.is_set():
        raise RuntimeError('已取消')
    try:
        with httpx.Client(transport=transport, timeout=httpx.Timeout(10, connect=5),
                          follow_redirects=False) as client:
            response = client.get('https://api.deepseek.com/user/balance',
                                  headers={'Authorization': 'Bearer ' + key})
        if cancel.is_set():
            raise RuntimeError('已取消')
        if response.status_code != 200:
            errors = {401: 'API Key 无效，请检查模型服务配置。',
                      403: '服务拒绝查询余额，请检查账户权限。',
                      429: '查询过于频繁，请稍后重试。'}
            raise RuntimeError(errors.get(response.status_code, '余额查询失败，请稍后重试。'))
        data = response.json()
        if not isinstance(data.get('is_available'), bool):
            raise ValueError
        amounts = {}
        for item in data['balance_infos']:
            currency = item['currency']
            raw = item['total_balance']
            if not isinstance(raw, str) or len(raw) > 64:
                raise ValueError
            amount = Decimal(raw)
            if (currency not in ('CNY', 'USD') or currency in amounts or not amount.is_finite()
                    or abs(amount) > Decimal('1000000000000')):
                raise ValueError
            amounts[currency] = amount
        if not amounts:
            raise ValueError
        return Balance(data['is_available'], amounts)
    except httpx.TimeoutException:
        raise RuntimeError('余额查询超时，请检查网络后重试。') from None
    except httpx.HTTPError:
        raise RuntimeError('无法连接余额服务，请检查网络。') from None
    except (ValueError, KeyError, TypeError, AttributeError, InvalidOperation):
        raise RuntimeError('服务返回的余额格式无效，请稍后重试。') from None


def balance_display(balance):
    symbols = {'CNY': '¥', 'USD': 'US$'}
    label = ' · '.join(f'{symbols[currency]}{amount:,.2f}'
                       for currency, amount in balance.amounts.items())
    return label, '' if balance.available else '账户当前不可调用'


class BalanceQuery:
    def __init__(self, events):
        self.events = events
        self.revision = 0
        self.cancel = threading.Event()
        self.checking = False

    def close(self):
        self.cancel.set()
        self.revision += 1
        self.checking = False

    def start(self, provider, profile):
        if self.checking:
            return
        self.close()
        self.cancel = threading.Event()
        revision, cancel = self.revision, self.cancel
        profile = dict(profile)
        self.checking = True

        def work():
            result, error = None, ''
            try:
                result = query_balance(provider, profile, cancel)
            except Exception as exc:
                error = str(exc) if isinstance(exc, RuntimeError) else '余额查询失败，请稍后重试。'
            if not cancel.is_set():
                self.events.put(('qa_balance', (revision, result, error)))
        threading.Thread(target=work, daemon=True, name='qa-balance').start()
