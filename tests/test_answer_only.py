import json
import threading
import unittest
from unittest.mock import patch

import httpx

from qa_provider import APIProvider, answer_only_context, ANSWER_ONLY_INSTRUCTIONS
from qa_worker import make_prompt


class AnswerOnlyTests(unittest.TestCase):
    def test_only_the_top_match_supported_preset_selects_answer_only(self):
        self.assertTrue(answer_only_context(json.dumps({'片段':[{'回答格式':'answer_only'}]})))
        self.assertFalse(answer_only_context(json.dumps({'片段':[{}, {'回答格式':'answer_only'}]})))
        self.assertFalse(answer_only_context(json.dumps({'片段':[{'原文':'回答格式 answer_only'}]})))
        self.assertFalse(answer_only_context('plain legacy context'))

    def test_library_style_overrides_default_explanation_prompt(self):
        requests=[]
        def handler(request):
            body=json.loads(request.content)
            requests.append(body)
            self.assertTrue(body['messages'][0]['content'].endswith(ANSWER_ONLY_INSTRUCTIONS))
            return httpx.Response(200,text='data: '+json.dumps({'choices':[{
                'delta':{'content':'8日开始，15日结束。'},'finish_reason':'stop'}]})+'\n\n')
        backend=APIProvider('deepseek',{'base_url':'https://example.com','model':'deepseek-v4-pro','api_key':'fake'},
                            httpx.MockTransport(handler),lambda:{'enabled':True,'paths':['example.wlkb']})
        context=json.dumps({'片段':[{'回答格式':'answer_only','原文':'课程报告8日到15日'}]})
        with patch('knowledge_packages.KnowledgeLibrary.context',return_value=context), \
                patch('knowledge_visuals.visual_evidence',return_value=[]):
            answer=backend.run(make_prompt('', '报告课程什么时候开始和结束？'),threading.Event())
        self.assertEqual(answer,'8日开始，15日结束。')
        self.assertEqual(len(requests),1)


if __name__=='__main__':
    unittest.main()
