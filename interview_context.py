"""Conservative topic selection for interview turns; never discard user facts."""
import re


ANSWER_CONTRACT = (
    '\n本轮作答约束：只回答当前问题，历史问答和资料只提供相关背景。'
    '默认口头回答控制在150至250个汉字、最多350字符，最多两个自然段，只讲两三个关键点。'
    '明确要求详细、代码、多题或特定格式时遵从该要求。'
    '经历题允许合理补全模拟设定并自然说成项目叙述，但不能与简历、明确的源码事实冲突；'
    '普通原理或排查题直接讲做法，不强行套项目。事实核验时严格区分证据和设定。'
)


def is_interview(prompt):
    return prompt.startswith('你正在代入使用者本人参加面试，')


def concise_interview(prompt, question):
    # Images may contain multiple questions or code not present in question text.
    return is_interview(prompt) and '图片识别结果' not in prompt and not re.search(
        r'详细|展开|完整|逐[题条项]|多[题个]|所有|每[个题]|代码(?!评审|审查)|算法|编程|实现一个|'
        r'列出|清单|对比表|表格|翻译|提取|原文|查证|核实|来源|引用|'
        r'\d+\s*(?:字|字符)|\b(?:code(?!\s*review)|implement|detailed|translate)\b', question, re.I)


_TOPICS = {
    'backend': r'高并发|并发|多线程|线程池|事务|慢\s*SQL|数据库|缓存|索引|秒杀|库存|订单|限流|熔断|幂等|分布式|分页|翻页|'
               r'列表.{0,20}翻.{0,12}慢|'
               r'\b(?:java|spring\s*boot|mybatis|mysql|redis|postgresql|jvm|hashmap)\b',
    'frontend': r'前端|小程序|响应式|白屏|首屏|事件循环|微任务|宏任务|组件|'
                r'\b(?:vue|react|javascript|typescript|css|html|taro|nexttick|promise)\b',
    'ai': r'代码评审|代码审查|评审流程|评审方案|大模型|知识库|向量|提示词|智能体|'
          r'\b(?:agent|rag|mcp|code\s*review|codereview|langchain|embedding|rerank)\b',
    'operations': r'部署|上线|回滚|运维|流水线|\b(?:docker|nginx|linux|ci/cd)\b',
}

# Explicit entities help a return to an older topic skip unrelated recent turns.
_ANCHORS = {
    'code_review': r'代码评审|代码审查|\bcode\s*review\b|\bcodereview\b',
    'rag': r'知识库|\brag\b',
    'mcp': r'\bmcp\b',
    'vue': r'\bvue\b',
    'react': r'\breact\b',
    'spring': r'\bspring(?:\s*boot)?\b',
    'nginx': r'\bnginx\b',
}


def anchors(question):
    return {name for name, pattern in _ANCHORS.items() if re.search(pattern, question, re.I)}


def topics(question):
    return {name for name, pattern in _TOPICS.items() if re.search(pattern, question, re.I)}


def needs_previous_question(question):
    """An explicit reference, not question length, justifies borrowing a topic."""
    question = question.strip().rstrip('？?。.!！')
    if re.search(r'刚才|上面|前面|上一[个轮次]|之前(?:说|提|讲)|'
                 r'(?:这个|那个|该|这种|这套|那套)[\w\s]{0,8}?(?:项目|系统|方案|流程|场景|问题|方法|接口|服务|工具)|'
                 r'第[一二三四五六七八九十\d]+[点步条个]|\b(?:that|above|previous)\b', question, re.I):
        return True
    if re.fullmatch(r'(?:嗯|好|好的|要的|可以|继续|继续讲|展开|详细点|举个例子|为什么|然后呢|还有呢|'
                    r'怎么做|怎么实现|怎么处理|有什么问题|有什么区别|能具体说说吗|再详细说说|yes|continue)',
                    question, re.I):
        return True
    if not topics(question):
        return bool(re.match(r'那(?:么)?|所以|具体|再|继续|它|这(?:个|样)|那(?:个|样)', question))
    return False


def history_scopes(questions):
    previous_topics, previous_anchors = set(), set()
    result = []
    for question in questions:
        current_topics, current_anchors = topics(question), anchors(question)
        if needs_previous_question(question) and not current_anchors:
            current_topics |= previous_topics
            current_anchors = previous_anchors
        previous_topics, previous_anchors = current_topics, current_anchors
        result.append((current_topics, current_anchors))
    return result


def retrieval_followups(question, questions):
    if not needs_previous_question(question):
        return []
    named = anchors(question)
    if named:
        scopes = history_scopes(questions)
        return [q for q, (_, names) in zip(questions, scopes) if names & named][-2:]
    return questions[-2:]


def select_topic_history(rows, exchanges, question):
    """Drop off-topic assistant examples only for explicit technical questions.

    Preserve transcripts and excluded user turns as background. Follow-ups inherit
    the last explicit topic when selecting a later independent question.
    """
    wanted = topics(question)
    named = anchors(question) if needs_previous_question(question) else set()
    if not wanted or (needs_previous_question(question) and not named):
        return rows, exchanges
    selected, user_background = [], []
    scopes = history_scopes([q for q, _ in exchanges])
    for (q, answer), (current, names) in zip(exchanges, scopes):
        relevant = bool(names & named) if named else (not current or bool(current & wanted))
        if relevant:
            selected.append((q, answer))
        else:
            user_background.append(('历史用户原话', q))
    return list(rows) + user_background, selected
