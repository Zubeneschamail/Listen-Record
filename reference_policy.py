"""Recognize explicit reference requests and unfinished promises to read them."""
import json
import re


_TOOL_MARKUP = re.compile(r'<\s*/?\s*[|｜\s]*DSML(?:[|｜\s]|$)', re.I)


def has_tool_markup(text):
    return isinstance(text, str) and (_TOOL_MARKUP.search(text) is not None or re.search(
        r'<\s*/?\s*[|｜][|｜\s]*(?:D(?:S(?:M(?:L)?)?)?)?\s*$', text, re.I) is not None)


def visible_partial(text):
    """Hold fragmented protocol prefixes until they can be classified."""
    marker = _TOOL_MARKUP.search(text)
    if marker:
        return text[:marker.start()]
    start = text.rfind('<')
    if start >= 0:
        suffix = re.sub(r'[|｜\s]', '', text[start:]).lower()
        if any(prefix.startswith(suffix) for prefix in ('<dsml', '</dsml')):
            return text[:start]
    return text


def clean_tool_history(history):
    cleaned = []
    for message in history or []:
        message = dict(message)
        if message.get('role') == 'assistant' and has_tool_markup(message.get('content')):
            message['content'] = '上一轮资料查阅返回了无效的工具指令，未完成查阅，不能据此认定已经读取资料。'
        cleaned.append(message)
    return cleaned


def reference_partial(text, scenario=False):
    """Hold an incomplete opening, tool protocol, and unexecuted read promises."""
    if has_tool_markup(text) or promises_reference_read(text, scenario):
        return ''
    visible = visible_partial(text)
    # Wait for a clause or a short opening so a fragmented "让我现在读取…"
    # cannot be displayed as an answer before it is recognized as a plan.
    if len(visible) < 32 and not re.search(r'[。！？\n]|[.!?](?:\s|$)', visible):
        return ''
    return visible


def current_question(text, depth=0):
    """Do not mistake quoted transcript/background instructions for the request."""
    if not isinstance(text, str) or depth > 4:
        return ''
    candidates = (text, text.partition('\n')[2])
    for candidate in candidates:
        try:
            value, _ = json.JSONDecoder().raw_decode(candidate.lstrip())
        except ValueError:
            continue
        if isinstance(value, dict) and isinstance(value.get('当前问题'), str):
            return value['当前问题']
        if isinstance(value, dict) and isinstance(value.get('原始请求'), str):
            return current_question(value['原始请求'], depth+1)
    return text


def asks_about_personal_experience(prompt):
    """Use the current question, never background transcripts, to select a resume."""
    question = current_question(prompt).strip()
    if re.search(r'(?:不要|不用|无需|不需要|别).{0,12}(?:简历|履历|资料|文件)', question):
        return False
    return bool(re.search(
        r'简历|履历|自我介绍|工作经历|工作经验|项目经历|职业经历|个人经历|'
        r'(?:你|我)在.{1,24}(?:做|负责|开发|工作|任职|项目|公司|团队|经历)|'
        r'(?:你|我)的(?:工作|项目|经历|职责|技术栈|背景|团队|公司)|'
        r'(?:你|我)(?:做过|负责|参与)|你们(?:项目|团队|公司)|'
        r'(?:介绍|讲讲|说说|谈谈|说下).{0,12}(?:自己|项目|工作|经历|职责)|'
        r'(?:做过|遇到过|处理过|实践|经验).{0,12}(?:并发|秒杀|缓存|事务|调优)|'
        r'(?:高并发|秒杀|库存超卖).{0,16}(?:怎么|如何|处理|设计|解决)|'
        r'(?:怎么|如何).{0,16}(?:高并发|秒杀|库存超卖)',
        question))


def promises_reference_read(answer, scenario=False):
    text = re.sub(r'```.*?```', '', answer, flags=re.S)
    if scenario:
        # In an interview, "我会先查日志" describes a troubleshooting method,
        # not a request for this application to access the machine. Still catch
        # actual promises to read the supplied references or delay the answer.
        text = '\n'.join(clause for clause in re.split(r'(?<=[。！？\n])', text)
                         if re.search(r'资料|参考|简历|知识包|README|稍等|等我|'
                                      r'\br\d+/|\b(?:reference|resume|wait)\b', clause, re.I))
    # Only first-person plans, not completed reads, user instructions or quotations.
    return bool(re.search(
        r'(?:^|[。！？\n])\s*(?:让我|我(?:现在|马上|这就|接下来|会|将|先|来)|'
        r'现在(?:就)?|马上|接下来)(?:就|先|来|去|会|将|再|直接|马上|现在|继续|帮你|为你|认真|尝试|实际|真正|\s)*'
        r'(?:读|查|看|检索|搜索)', text)
        or re.search(r"\bI(?:'ll| will| am going to)\s+(?:now\s+)?(?:read|search|inspect|check)\b", text, re.I))


def requires_reference_tools(prompt, history=None):
    _, separator, user = prompt.partition('\n')
    question = current_question(user if separator and not prompt.lstrip().startswith('{') else prompt).strip()
    if re.search(r'(?:不要|不用|无需|别).{0,8}(?:读|查|搜索|检索)|\b(?:do not|don.t)\s+(?:read|search)', question, re.I):
        return False
    target = r'资料|文件|目录|项目|仓库|源码|代码库|README|\b(?:file|folder|directory|repository|repo|project)\b|\br\d+/|[\w-]+\.(?:ts|js|py|json|md|txt|cpp|cs)\b'
    action = r'读|查|看|搜索|检索|分析|介绍|总结|概括|结构|内容|实现|什么|哪|\b(?:read|search|inspect|list|analy[sz]e|summarize|find|show)\b'
    if re.search(target, question, re.I) and re.search(action, question, re.I):
        return True
    if re.fullmatch(r'(?:读|看|查)(?:完|过|好)?了[吗么没]?|继续|继续读|现在读|读一下|好的?|要的|可以',
                    question.rstrip('？?！!。.'), re.I):
        for message in (history or [])[-4:]:
            text = current_question(message.get('content', ''))
            if (re.search(target, text, re.I) and re.search(action, text, re.I)) or promises_reference_read(text):
                return True
    return False
