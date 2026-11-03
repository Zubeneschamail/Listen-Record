"""Opt-in live smoke test using the user's configured provider and references.

Run manually only: consumes API credits. Never prints credentials.
"""
import json
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qa_provider import APIProvider, load_settings
from reference_files import load_settings as load_references
from qa_worker import make_prompt
from session_context import build_conversation
from knowledge_packages import make_query


def main():
    destination = Path(sys.argv[1])
    settings = load_settings()
    provider = settings['provider']
    profile = settings['profiles'][provider]
    backend = APIProvider(provider, profile, workspace_loader=load_references)
    report = {'provider': provider, 'model': profile['model'], 'cases': []}

    def ask(label, question, exchanges):
        rows = [('system', q) for q, _ in exchanges]
        background, history = build_conversation(rows, exchanges, question)
        prompt = make_prompt('', question, background)
        usage = []
        start = time.monotonic()
        print('START ' + label, flush=True)
        answer = backend.run(prompt, threading.Event(), timeout=120, history=history,
                             usage=lambda item: usage.append(item))
        case = {'case': label, 'question': question, 'answer': answer,
                'elapsed_seconds': round(time.monotonic()-start, 2),
                'history_questions': [m['content'] for m in history if m['role'] == 'user'],
                'retrieval_query': make_query(prompt, history), 'usage': usage}
        report['cases'].append(case)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(case, ensure_ascii=False), flush=True)
        return question, answer

    print(json.dumps({'provider': provider, 'model': profile['model']}, ensure_ascii=False), flush=True)
    # Reuse completed live seed turns if an interrupted run already saved them.
    if destination.exists():
        saved = json.loads(destination.read_text(encoding='utf-8'))
        report['cases'] = [c for c in saved['cases'] if c['case'] in ('agent_project', 'review_followup')]
    seed = [(c['question'], c['answer']) for c in report['cases']]
    if not seed:
        seed.append(ask('agent_project', '介绍一下你做的 Agent 和 Code Review 项目。', []))
    if len(seed) == 1:
        seed.append(ask('review_followup', '这套评审流程怎么接入研发工作流？', seed))
    for label, question in (
        ('frontend', '你做过的页面首屏比较慢时，是怎么定位和优化的？'),
        ('transaction', '事务明明加了注解，更新出错后却没有回滚，你会怎么排查？'),
        ('sql', '列表数据多了以后翻到后面越来越慢，这种情况怎么处理？'),
        ('deployment', '服务上线后 Nginx 返回 502，你会怎么定位？'),
    ):
        seed.append(ask(label, question, seed))
    ask('explicit_review_return', '回到刚才的 Code Review，如果 Jira 接口超时，你会怎么处理？', seed)


if __name__ == '__main__':
    main()
