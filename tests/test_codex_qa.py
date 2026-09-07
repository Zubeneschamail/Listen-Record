import json
import queue
import threading
import unittest
from codex_qa import CodexQA, command, is_question


class CodexQATests(unittest.TestCase):
    def test_question_detection_and_debounce_with_context(self):
        now = [10.0]
        events = queue.Queue()
        prompts = []
        def run(prompt, cancel):
            prompts.append(prompt)
            return "示例答案"
        qa = CodexQA(events, run, lambda: now[0])
        qa.feed("关闭时不能发送什么？")
        self.assertFalse(qa.context)
        qa.set_enabled(True)
        qa.feed("我们正在讨论机器学习。")
        qa.feed("什么是")
        qa.tick()
        self.assertTrue(events.empty())
        now[0] += 0.5
        qa.feed("大模型？")
        now[0] += 2
        qa.tick()
        self.assertEqual(events.get(timeout=2)[1][1], "thinking")
        result = events.get(timeout=2)[1]
        self.assertEqual(result[1], "answer")
        self.assertEqual(result[2], ("什么是\n大模型？", "示例答案"))
        data = json.loads(prompts[0].split("\n", 1)[1])
        self.assertIn("机器学习", data["背景转写"])
        self.assertEqual(data["当前问题"], "什么是\n大模型？")
        self.assertTrue(is_question("Whisper 是免费的吗。"))
        self.assertFalse(is_question("这是一个语音模型。"))

    def test_cancellation_drops_answer_and_new_generation(self):
        events, started, release = queue.Queue(), threading.Event(), threading.Event()
        def run(prompt, cancel):
            started.set()
            release.wait(2)
            return "旧答案"
        qa = CodexQA(events, run)
        qa.set_enabled(True)
        qa.feed("为什么？")
        qa.tick(force=True)
        self.assertTrue(started.wait(2))
        old = qa.generation
        qa.set_enabled(False)
        release.set()
        self.assertEqual(events.get(timeout=2)[1][1], "thinking")
        result = events.get(timeout=2)[1]
        self.assertEqual(result[:2], (old, "done"))
        self.assertNotEqual(old, qa.generation)
        self.assertFalse(qa.active)
        self.assertFalse(qa.pending)

    def test_duplicate_question_and_bounded_context(self):
        events = queue.Queue()
        qa = CodexQA(events, lambda p, c: "答案")
        qa.set_enabled(True)
        qa.feed("为什么天空是蓝色？")
        qa.tick(force=True)
        for _ in range(3):
            events.get(timeout=2)
        qa.active = False
        qa.feed("为什么天空是蓝色？")
        qa.last_input -= 10
        qa.tick()
        self.assertTrue(events.empty())
        for _ in range(30):
            qa.feed("背景。" * 1000)
        self.assertEqual(len(qa.context), 12)
        self.assertLessEqual(len(qa.context[-1]), 1200)

    def test_cli_uses_isolated_working_directory_and_no_shell_tools(self):
        args = command("codex.exe", "C:/temp/qa")
        self.assertIn("--ignore-user-config", args)
        self.assertIn("--ephemeral", args)
        self.assertIn("read-only", args)
        self.assertEqual(args[args.index("-C") + 1], "C:/temp/qa")
        self.assertEqual(args[args.index("shell_tool") - 1], "--disable")
        self.assertEqual(args[-1], "-")


if __name__ == "__main__":
    unittest.main()
