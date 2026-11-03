import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch

import httpx
from PIL import Image

from knowledge_packages import make_query
from knowledge_visuals import FORMAT, visual_evidence
from qa_provider import APIProvider, image_analysis_prompt
from qa_worker import make_prompt


def jpeg():
    buffer = io.BytesIO()
    Image.new('RGB', (12, 12), 'green').save(buffer, format='JPEG')
    return buffer.getvalue()


class KnowledgeVisualTests(unittest.TestCase):
    def test_only_images_linked_to_the_exact_retrieved_chunk_are_read(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)/'test.wlkb'
            package.touch()
            image = jpeg()
            def opened(*args):
                connection = sqlite3.connect(':memory:')
                connection.row_factory = sqlite3.Row
                connection.executescript(
                    'CREATE TABLE kb_visual_pages(document TEXT,page INTEGER,image BLOB,sha256 TEXT);'
                    'CREATE TABLE kb_chunk_visuals(source TEXT,line_start INTEGER,line_end INTEGER,document TEXT,page INTEGER);')
                connection.execute('INSERT INTO kb_visual_pages VALUES(?,?,?,?)',
                                   ('PDF2',8,image,hashlib.sha256(image).hexdigest()))
                connection.execute('INSERT INTO kb_chunk_visuals VALUES(?,?,?,?,?)',('questions.txt',10,30,'PDF2',8))
                return connection, {'name':'题库','visual_attachment_format':FORMAT}
            settings = {'enabled':True,'paths':[str(package)]}
            match = {'来源':'kb1/test.wlkb/questions.txt','起始行':10,'结束行':30}
            with patch('knowledge_visuals._open',side_effect=opened):
                actual = visual_evidence(settings,json.dumps({'片段':[match,match]}))
                self.assertEqual(len(actual),1)
                self.assertEqual(actual[0]['data'],image)
                self.assertIn('PDF2原第8页',actual[0]['label'])
                wrong = dict(match,结束行=31)
                self.assertEqual(visual_evidence(settings,json.dumps({'片段':[wrong]})),[])
                outside = dict(match,来源='kb1/test.wlkb/../../secret.jpg')
                self.assertEqual(visual_evidence(settings,json.dumps({'片段':[outside]})),[])

    def test_image_question_keeps_tail_constraints_in_retrieval_query(self):
        description = '课程时间表，数据收集三天。'+'很长的表格描述。'*60+'报告课程开始和结束日期，29日至30日考试。'
        prompt = image_analysis_prompt(make_prompt('', '请回答图中问题'),description)
        question, _ = make_query(prompt,[])
        self.assertIn('数据收集',question)
        self.assertIn('报告课程开始和结束日期',question)
        self.assertIn('29日至30日',question)

    def test_pro_receives_selected_dates_read_from_embedded_reference_images(self):
        requests = []
        def response(content):
            return httpx.Response(200,text='data: '+json.dumps({'choices':[{'delta':{'content':content},'finish_reason':'stop'}]})+'\n\n')
        def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            if len(requests)==1:
                self.assertEqual(body['model'],'deepseek-flash')
                return response('课程时间表：数据收集三天，数据清理最短，数据报告最长。问报告开始和结束日期。')
            if len(requests) in (2,3):
                self.assertEqual(body['model'],'deepseek-flash')
                image_parts = [p for p in body['messages'][-1]['content'] if p['type']=='image_url']
                self.assertEqual(len(image_parts),1)
                self.assertTrue(all(p['image_url']['url'].startswith('data:image/jpeg;base64,') for p in image_parts))
                return response('同一道题，日历1日周一，答案绿色圈选8日、15日，分别为开始和结束。'
                                if len(requests)==2 else '本页为其他题目，不匹配当前课程题。')
            self.assertEqual(body['model'],'deepseek-v4-pro')
            content = body['messages'][-1]['content']
            self.assertIsInstance(content,str)
            self.assertIn('绿色圈选8日、15日',content)
            return response('8日开始，15日结束。')
        backend = APIProvider('deepseek',{'base_url':'https://example.com','model':'deepseek-v4-pro','api_key':'fake'},
                              httpx.MockTransport(handler),lambda:{'enabled':True,'paths':['example.wlkb']})
        refs = [{'label':f'PDF2原第{page}页','mime':'image/jpeg','data':jpeg()} for page in (8,12)]
        with patch('knowledge_packages.KnowledgeLibrary.context',return_value='{"片段":[]}'), \
                patch('knowledge_visuals.visual_evidence',return_value=refs):
            answer = backend.run(make_prompt('', '请回答图片问题'),threading.Event(),image=b'input-png')
        self.assertEqual(answer,'8日开始，15日结束。')
        self.assertEqual(len(requests),4)


if __name__=='__main__':
    unittest.main()
