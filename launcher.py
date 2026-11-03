"""Executable entry point, with a no-audio packaged smoke check."""
import json
from pathlib import Path
import sys


def verify_pending_words():
    import queue
    from types import SimpleNamespace
    import numpy as np
    from app import Transcriber
    from paragraphs import ParagraphAssembler
    from recognition import RecognitionContext
    from segmentation import AudioChunk, WordAssembler, RATE
    events = queue.Queue()
    engine = Transcriber(events)
    assembler = WordAssembler()
    first = AudioChunk(np.zeros(15*RATE, np.float32), 0., 1800000000., forced=True)
    output = assembler.consume(first, [SimpleNamespace(words=[
        SimpleNamespace(start=.2, end=14.3, word='密等性'),
        SimpleNamespace(start=14.7, end=14.9, word='未确认')])])
    if output or assembler.pending is None:
        raise RuntimeError('Packaged hard-cut prefix was not held')
    states = dict(system=dict(assembler=assembler, guidance=RecognitionContext('密等性=幂等性')))
    paragraphs = ParagraphAssembler()
    engine._finish_pending_words(states, paragraphs)
    engine._finish_pending_words(states, paragraphs)
    for paragraph in paragraphs.finish():
        engine.emit('segment', paragraph)
    messages = list(events.queue)
    if (len(messages) != 1 or messages[0][0] != 'segment' or
            messages[0][1]['text'] != '幂等性' or messages[0][1]['source'] != 'system' or
            messages[0][1].get('incomplete') is not True or
            messages[0][1]['start'] != .2 or messages[0][1]['end'] != 14.3 or
            assembler.pending is not None or assembler.finish()):
        raise RuntimeError('Packaged stable-prefix recovery lost or duplicated words')
    return dict(stable_words_cleanup=True)


def verify_capture_loss_tails():
    import queue
    import time
    from types import SimpleNamespace
    from app import Transcriber
    from paragraphs import ParagraphAssembler
    for incomplete in (False, True):
        paragraphs = ParagraphAssembler()
        paragraphs.consume(dict(start=0., end=1., text='缓冲的前半句', source='system'))
        emitted = []
        states = {
            'system': {'segmenter': SimpleNamespace(finish=lambda: [])},
            'microphone': {'segmenter': SimpleNamespace(finish=lambda: [dict(start=1., end=2., text='另一音源的尾句')])},
        }
        def transcribe(source, rows, *, incomplete=False):
            for original in rows:
                row = dict(original, source=source)
                if incomplete:
                    row['incomplete'] = True
                emitted.extend(paragraphs.consume(row))
        Transcriber(queue.Queue())._finish_sources(states, transcribe, paragraphs, incomplete=incomplete)
        emitted.extend(paragraphs.finish())
        if (len(emitted) != 2 or
                any(row.get('incomplete', False) != incomplete for row in emitted)):
            raise RuntimeError('Packaged capture loss did not mark all unfinished source tails')
        paragraphs = ParagraphAssembler()
        paragraphs.consume(dict(start=0., end=1., text='等待刷出的尾句', source='microphone'))
        emitted = []
        states = {'system': {'received': time.monotonic()-2.,
            'segmenter': SimpleNamespace(finish=lambda: [dict(start=1., end=2., text='空闲音源的尾句')])}}
        events = queue.Queue()
        Transcriber(events)._finish_idle_sources(states, transcribe, paragraphs,
            lambda source, active: None, incomplete=incomplete)
        emitted.extend(row for kind, row in events.queue if kind == 'segment')
        emitted.extend(paragraphs.finish())
        if (len(emitted) != 2 or
                any(row.get('incomplete', False) != incomplete for row in emitted)):
            raise RuntimeError('Packaged idle source flush lost capture interruption markers')
    return dict(capture_loss_tail_guard=True)


def verify_incomplete_voice():
    import queue
    import tempfile
    from types import SimpleNamespace
    from app import App, export_text, save_segment
    from qa_worker import QAWorker
    from transcript_corrections import load_recording
    events = queue.Queue()
    worker = QAWorker(events, clock=lambda: 10.)
    worker.set_enabled(True)
    worker.feed('什么是 RAG？', record_key=0)
    worker.collect_voice(force=True)
    worker.feed('请比较一下微调在', record_key=1)
    worker.feed('某个场景下', record_key=2, complete=False)
    worker.feed('为什么向量检索', record_key=3)
    worker.interrupt_voice()
    if (worker.pending or len(worker.voice_queue) != 1 or
            worker.voice_queue[0].question.text() != '什么是 RAG？' or
            '某个场景下' in worker.context):
        raise RuntimeError('Packaged interrupted voice was queued or erased a complete question')
    row = dict(start=0., end=1., text='请比较一下微调在', source='system', incomplete=True)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)/'interrupted.jsonl'
        save_segment(path, row)
        restored = load_recording(path).rows
        label = App.transcript_time(SimpleNamespace(corrected_rows=set()), 0, row)
        if (restored != [row] or '转录中断' not in label or
                '转录中断' not in export_text(restored) or
                '转录中断' not in path.with_suffix('.txt').read_text('utf-8-sig')):
            raise RuntimeError('Packaged interrupted transcript lost its saved marker')
    return dict(incomplete_voice_guard=True, incomplete_marker_roundtrip=True)


def verify_saved_wordlists():
    import tempfile
    from recognition import load_capture_preferences, save_preferences, RecognitionContext
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)/'recognition-settings.json'
        if load_capture_preferences(path) or path.exists():
            raise RuntimeError('Capture created an absent word-list file')
        path.write_text(json.dumps({'hotwords':'密等性=幂等性, RAG','note':{'keep':True}}, ensure_ascii=False), encoding='utf-8')
        original = path.read_bytes()
        value = load_capture_preferences(path)
        if path.read_bytes() != original or RecognitionContext(value).normalize('密等性') != '幂等性':
            raise RuntimeError('Capture did not read saved rules without writing')
        save_preferences(path, 'RAG')
        if json.loads(path.read_text(encoding='utf-8'))['note'] != {'keep':True}:
            raise RuntimeError('Explicit word-list edit lost unrelated fields')
        path.write_bytes(b'{broken')
        try:
            load_capture_preferences(path)
        except ValueError:
            if path.read_bytes() != b'{broken':
                raise RuntimeError('Capture modified an invalid word list')
        else:
            raise RuntimeError('Capture silently ignored an invalid saved word list')
    return dict(capture_wordlist_read_only=True, capture_wordlist_strict=True,
                wordlist_metadata_preserved=True)


def verify_live_corrections():
    from copy import deepcopy
    from recognition import CorrectionUpdates, RecognitionContext
    updates = CorrectionUpdates()
    contexts = [RecognitionContext('RAG', correction_source=updates.snapshot) for _ in range(2)]
    before = [context.options(0) for context in contexts]
    updates.publish('密等性=幂等性, PostgreSQL')
    if any(context.normalize('密等性') != '幂等性' for context in contexts):
        raise RuntimeError('Saved corrections did not reach both capture contexts')
    if [context.options(0) for context in contexts] != before:
        raise RuntimeError('Live corrections changed acoustic hints')
    draft = deepcopy(contexts[0])
    updates.publish('密等性=新规则')
    if draft.normalize('密等性') != '新规则' or draft.correction_source.__self__ is not updates:
        raise RuntimeError('Draft context did not retain live correction updates')
    cold = RecognitionContext('a gent=Agent, agent')
    if cold.normalize('a gent') != 'Agent':
        raise RuntimeError('Saved correction spelling was overridden after restart')
    return dict(live_saved_corrections=True, live_rule_hints_preserved=True, live_rule_preview_copy=True)


def verify_segmentation_policy():
    """Verify the actual frozen app's backend policy without recording devices."""
    import numpy as np
    from app import create_segmenter
    from qwen_engine import QWEN_MODEL
    from segmentation import RATE

    def detector(audio):
        edges = np.flatnonzero(np.diff(np.pad((audio > .5).astype(np.int8), (1, 1))))
        return [dict(start=int(start), end=int(end)) for start, end in zip(edges[::2], edges[1::2])]

    def feed(name, audio):
        state = create_segmenter(name)
        state.detector = detector
        chunks = []
        for offset in range(0, len(audio), RATE // 10):
            chunks.extend(state.push(audio[offset:offset + RATE // 10], offset / RATE, 1000. + offset / RATE))
        return chunks + state.finish()

    audio = np.zeros(3 * RATE, dtype=np.float32)
    audio[:round(.35 * RATE)] = 1.
    audio[round(.99 * RATE):round(1.8 * RATE)] = 1.
    expected = np.count_nonzero(audio)
    for name in (QWEN_MODEL, 'small', 'large-v3-turbo', 'base', 'tiny'):
        chunks = feed(name, audio)
        if (len(chunks) != (1 if name == QWEN_MODEL else 2) or
                sum(np.count_nonzero(chunk.audio) for chunk in chunks) != expected):
            raise RuntimeError('Packaged model-specific speech segmentation failed')
    short = np.ones(round(.35 * RATE), dtype=np.float32)
    chunks = feed(QWEN_MODEL, short)
    if len(chunks) != 1 or not np.array_equal(chunks[0].audio, short):
        raise RuntimeError('Packaged short reply stop/flush failed')
    return dict(qwen_short_onset_context=True, whisper_pause_policy_preserved=True,
                short_reply_stop_flush=True)


def verify_capture_owner(device='auto'):
    """Exercise capture-model handoff using the actual packaged Whisper runtime."""
    import threading
    import numpy as np
    from model_runtime import load_capture_model, release_model
    from whisper_runtime import WhisperCaptureModel
    loaded, release = threading.Event(), threading.Event()
    holder, errors, clients = {}, [], []
    def create():
        clients.append(threading.get_ident())
        try:
            holder['model'] = load_capture_model('small', device=device)
        except Exception as exc:
            errors.append(str(exc))
        finally:
            loaded.set()
        release.wait()
    creator = threading.Thread(target=create, daemon=True)
    creator.start()
    model = None
    try:
        if not loaded.wait(timeout=120):
            raise TimeoutError('Capture model did not finish loading')
        if errors:
            raise RuntimeError(errors[0])
        model = holder['model']
        if not isinstance(model, WhisperCaptureModel):
            raise RuntimeError('Capture model is missing its persistent owner')
        backend = model.model.device
        owner = model._thread.ident
        def infer():
            clients.append(threading.get_ident())
            try:
                list(model.transcribe(np.zeros(16000, dtype=np.float32), language='zh',
                    beam_size=1, vad_filter=False, max_new_tokens=1)[0])
            except Exception as exc:
                errors.append(str(exc))
        decoder = threading.Thread(target=infer, daemon=True)
        decoder.start(); decoder.join(timeout=120)
        if decoder.is_alive():
            raise TimeoutError('Capture-model inference did not finish')
        if errors:
            raise RuntimeError(errors[0])
        if len(set(clients)) != 2 or owner in clients:
            raise RuntimeError('Capture ownership verification used the wrong threads')
        release.set(); creator.join(timeout=2)
        release_model(model)
        if model._thread.is_alive() or not model.closed:
            raise RuntimeError('Capture-model owner did not exit')
        return dict(backend=backend, distinct_capture_clients=True, persistent_owner=True,
                    owner_thread_exited=True, tokenizer_budget_available=model.hf_tokenizer is not None)
    finally:
        release.set()
        creator.join(timeout=2)
        if model is not None:
            release_model(model)


def verify_optional_qwen(audio_path=None):
    """Exercise the installed engine from the frozen desktop host, offline."""
    import ctypes
    from ctypes import wintypes
    import hashlib
    import numpy as np
    from model_runtime import load_model, release_model
    from qwen_engine import QWEN_MODEL, engine_root
    root = engine_root()
    if root is None:
        raise RuntimeError('Optional Qwen component is missing or incomplete')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    model = handle = None
    result = dict(engine=str(root), gpu_warmup=True)
    try:
        model = load_model(QWEN_MODEL, device='cuda')
        handle = kernel.OpenProcess(0x00100000, False, model.worker_pid)
        if not handle:
            raise RuntimeError('Cannot monitor the actual engine process')
        if list(model.transcribe(np.zeros(16000, dtype=np.float32), language='zh')[0]) or model._sequence:
            raise RuntimeError('Silence was submitted to the optional recognizer')
        result['silence_rejected'] = True
        if audio_path is not None:
            from faster_whisper.audio import decode_audio
            from audio_levels import AudioLevelNormalizer
            from app import clean_caption
            audio_path = Path(audio_path)
            raw = decode_audio(str(audio_path), sampling_rate=16000)
            levels = AudioLevelNormalizer()
            sound = np.concatenate([levels.process(raw[i:i+1600]) for i in range(0, len(raw), 1600)])
            segments = list(model.transcribe(sound, language='zh', word_timestamps=True)[0])
            result.update(audio_sha256=hashlib.sha256(audio_path.read_bytes()).hexdigest(),
                samples_sha256=hashlib.sha256(sound.tobytes()).hexdigest(),
                text=clean_caption(''.join(s.text for s in segments)),
                words=[vars(word) for segment in segments for word in segment.words])
        release_model(model)
        if kernel.WaitForSingleObject(handle, 0) != 0:
            raise RuntimeError('Actual optional engine process did not exit')
        result['actual_worker_exited'] = True
        return result
    finally:
        if model is not None:
            release_model(model)
        if handle:
            kernel.CloseHandle(handle)


def verify_storage_stop_qa(app):
    """Use the shipped App policies without devices, Tk, or external requests."""
    import queue
    from types import SimpleNamespace
    from qa_worker import QAWorker

    class Value:
        def __init__(self, value=None): self.value = value
        def get(self): return self.value
        def set(self, value): self.value = value

    qa = QAWorker(queue.Queue(), clock=lambda: 10.)
    qa.set_enabled(True)
    qa.feed('什么是 RAG？', record_key=0)
    qa.collect_voice(force=True)
    qa.feed('请比较一下 RAG 和微调在', record_key=1)
    qa.active, qa.source = True, 'manual'
    generation, cancel = qa.generation, qa.cancel
    stopped = []
    owner = SimpleNamespace(qa=qa, auto_qa=Value(True), session_rows=[{}, {}],
        engine=SimpleNamespace(stop=lambda: stopped.append(True)),
        status=Value(), footer=Value(), qa_status=Value(), session_save_failed=False, failed=False)
    app.App.recording_save_error(owner, OSError('self-test write failure'))
    previous_context = list(qa.context)
    app.App.feed_automatic_voice(owner, dict(text='部署时'), 2)
    if (not stopped or not owner.session_save_failed or qa.pending or len(qa.voice_queue) != 1 or
            list(qa.context) != previous_context or not qa.active or qa.source != 'manual' or
            qa.generation != generation or qa.cancel is not cancel or cancel.is_set()):
        raise RuntimeError('Packaged storage-stop Q&A boundary failed')
    owner.session_save_failed = False
    app.App.feed_automatic_voice(owner, dict(text='为什么需要向量检索？'), 3)
    if not qa.pending:
        raise RuntimeError('Packaged next-recording voice input did not resume')
    return dict(storage_stop_qa_guard=True)


def verify_append_recovery(app):
    """A reported partial write must not invalidate an earlier saved record."""
    import tempfile
    from qa_records import append_qa, load_qa
    from transcript_corrections import read_rows

    class PartialFailure:
        def __init__(self, stream): self.stream = stream
        def __getattr__(self, name): return getattr(self.stream, name)
        def __enter__(self): self.stream.__enter__(); return self
        def __exit__(self, *args): return self.stream.__exit__(*args)
        def write(self, value):
            self.stream.write(value[:max(1, len(value)//2)])
            raise OSError('self-test partial write')

    with tempfile.TemporaryDirectory(prefix='wenlu-append-check-') as folder:
        for kind in ('transcript', 'qa', 'txt'):
            path = Path(folder)/(kind+'.jsonl')
            if kind == 'qa':
                action = lambda text: append_qa(path, text, 'local question', text)
                reader = load_qa
            else:
                action = lambda text: app.save_segment(path, dict(start=0, end=1, text=text))
                reader = read_rows
            action('previous complete record')
            target = path.with_suffix('.txt') if kind == 'txt' else path
            previous = target.read_bytes()
            original_open = Path.open
            def opened(actual, *args, **kwargs):
                stream = original_open(actual, *args, **kwargs)
                return PartialFailure(stream) if actual == target and args and 'a' in args[0] else stream
            Path.open = opened
            try:
                try:
                    action('interrupted new record')
                except OSError:
                    pass
                else:
                    raise RuntimeError('Packaged write fault was not reported')
            finally:
                Path.open = original_open
            if target.read_bytes() != previous or len(reader(path)) != (2 if kind == 'txt' else 1):
                raise RuntimeError('Packaged append recovery damaged saved history')
    return dict(record_append_rollback=True)


def verify_recording_io(app):
    """Check shipped record validation/exports using only temporary public text."""
    import codecs
    import tempfile
    from transcript_corrections import load_recording, row_origin, save_correction, corrections_path
    row = dict(start=1.25, end=3.75, text='什么是密等性？', source='system',
               captured_at='2026-11-02T10:00:00+08:00')
    with tempfile.TemporaryDirectory(prefix='wenlu-record-check-') as folder:
        path = Path(folder)/'practice.jsonl'
        raw = codecs.BOM_UTF8+(json.dumps(row, ensure_ascii=False)+'\n').encode('utf-8')
        path.write_bytes(raw)
        save_correction(row_origin(path, 0, row), '什么是幂等性？')
        sidecar = corrections_path(path)
        sidecar.write_bytes(codecs.BOM_UTF8+sidecar.read_bytes())
        document = load_recording(path)
        if document.rows[0]['text'] != '什么是幂等性？' or document.corrected != {0} or document.warnings:
            raise RuntimeError('Packaged BOM recording/correction recovery failed')
        for change in (dict(source=3), dict(end=None), dict(start=float('nan')), dict(end=float('inf'))):
            bad = Path(folder)/'bad.jsonl'
            bad.write_text(json.dumps({**row, **change})+'\n', encoding='utf-8')
            try:
                load_recording(bad)
            except ValueError as exc:
                if '第 1 行' not in str(exc):
                    raise RuntimeError('Packaged record error lost its line number') from exc
            else:
                raise RuntimeError('Packaged invalid recording was accepted')
        txt = path.with_suffix('.txt')
        txt.write_bytes(b'original auto-saved transcript')
        for target in (path, txt, sidecar):
            before = target.read_bytes()
            try:
                app.save_transcript_export(target, document.rows, source_paths=(path, txt))
            except ValueError:
                pass
            else:
                raise RuntimeError('Packaged export overwrote a practice source')
            if target.read_bytes() != before:
                raise RuntimeError('Packaged rejected export changed its target')
        for extension in ('txt', 'SRT'):
            target = Path(folder)/('export.'+extension)
            target.write_bytes(b'previous export')
            app.save_transcript_export(target, document.rows)
            if (target.read_text(encoding='utf-8-sig') != app.export_text(document.rows, extension == 'SRT')
                    or target.read_bytes().count(codecs.BOM_UTF8) != 1):
                raise RuntimeError('Packaged corrected transcript export failed')
        if path.read_bytes() != raw or list(Path(folder).glob('*.tmp')):
            raise RuntimeError('Packaged record recovery left changed sources or temporary files')
    return dict(recording_import_validation=True, utf8_bom_recording=True,
                transcript_export_guard=True, atomic_transcript_export=True)


def verify_interview_runtime(app):
    """Exercise shipped interview behavior without capture, settings or API calls."""
    import queue
    import tempfile
    import numpy as np
    from recognition import RecognitionContext, INTERVIEW_HOTWORDS
    from qa_worker import QAWorker, is_question
    from segmentation import AudioChunk, RATE, WordAssembler, PauseSegmenter, budget_prompt
    from tokenizers import Tokenizer, models, pre_tokenizers
    from types import SimpleNamespace
    canonical = RecognitionContext('C++, .NET, 密等性=幂等性').normalize('c + +、. net、密等性')
    if canonical != 'C++、.NET、幂等性' or len(INTERVIEW_HOTWORDS) != 4:
        raise RuntimeError('Packaged interview terminology failed')
    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Split('', behavior='isolated')
    bounded = budget_prompt(SimpleNamespace(hf_tokenizer=tokenizer, max_length=285),
                            dict(hotwords='LangGraph, PostgreSQL'))
    long_target = 'VeryLongTechnicalIdentifierForInterviewTermCanonicalCase'
    if (bounded['hotwords'] != 'LangGraph' or
            RecognitionContext('旧称='+long_target).normalize('旧称') != long_target):
        raise RuntimeError('Packaged complete hotword/alias preservation failed')
    wave = np.sin(np.arange(1600, dtype=np.float32)*.1)*.2
    if not np.allclose(app.mono_16k(np.column_stack([wave, -wave]), 16000), wave, atol=1e-6):
        raise RuntimeError('Packaged stereo speech protection failed')
    events, clock = queue.Queue(), [0.]
    qa = QAWorker(events, lambda prompt, cancel: '本地打包验证', lambda: clock[0])
    qa.set_enabled(True)
    qa.set_question_source('system')
    question = '为什么使用消息队列？'
    qa.feed(question, 'system')
    qa.feed('我的项目里负责接口。', 'microphone')
    qa.observe_voice('system', True)
    clock[0] = 20
    qa.tick()
    if qa.active or not events.empty():
        raise RuntimeError('Packaged Q&A submitted while interviewer was speaking')
    qa.observe_voice('system', False)
    qa.tick()
    phases = [events.get(timeout=5)[1] for _ in range(3)]
    if phases[0][1:] != ('thinking', question) or phases[1][1] != 'answer':
        raise RuntimeError('Packaged interviewer source routing failed')
    qa.active = False
    qa.feed(question, 'system')
    clock[0] += 2
    qa.tick()
    if not events.empty():
        raise RuntimeError('Packaged duplicate transcript suppression failed')
    qa.feed('因为需要异步处理订单。', 'microphone')
    qa.feed(question, 'system')
    clock[0] += 2
    qa.tick()
    phases = [events.get(timeout=5)[1] for _ in range(3)]
    if phases[0][1:] != ('thinking', question):
        raise RuntimeError('Packaged contextual repeated question failed')
    if not all(is_question(text) for text in ('举个具体例子。', '设计一个订单系统。', '手写一个LRU缓存。')):
        raise RuntimeError('Packaged spoken interview commands failed')
    qa.active = False
    qa.feed('再具体一点。')
    clock[0] += 2
    qa.tick()
    phases = [events.get(timeout=5)[1] for _ in range(3)]
    qa.set_enabled(False)
    if phases[0][1:] != ('thinking', '再具体一点。'):
        raise RuntimeError('Packaged short follow-up failed')
    rows = [dict(start=0, end=1, text='打包前段'), dict(start=1, end=2, text='打包后段')]
    with tempfile.TemporaryDirectory(prefix='wenlu-package-check-') as folder:
        path = Path(folder)/'recording.jsonl'
        for row in rows:
            app.save_segment(path, row)
        restored = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        if restored != rows or path.with_suffix('.txt').read_text(encoding='utf-8-sig') != app.export_text(rows):
            raise RuntimeError('Packaged recording save failed')
        from transcript_corrections import load_recording, row_origin, save_correction
        original = path.read_bytes(), path.with_suffix('.txt').read_bytes()
        origin = row_origin(path, 0, rows[0])
        save_correction(origin, '用户修正后的转录')
        document = load_recording(path)
        if (document.rows[0]['text'] != '用户修正后的转录' or document.corrected != {0} or
                original != (path.read_bytes(), path.with_suffix('.txt').read_bytes())):
            raise RuntimeError('Packaged original-preserving transcript correction failed')
        save_correction(origin, rows[0]['text'])
        if load_recording(path).rows != rows:
            raise RuntimeError('Packaged original transcript restoration failed')
        from recognition_rules import load_rule_preferences, save_rule
        from recognition import read_preferences
        settings = Path(folder)/'recognition-settings.json'
        settings.write_text(json.dumps({'hotwords': 'lang chain=LangGraph, C++, .NET'}), encoding='utf-8')
        state = load_rule_preferences(settings)
        value = save_rule(settings, '密等性', '幂等性', state.fingerprint)
        context = RecognitionContext(read_preferences(settings))
        if (context.normalize('密等性和密钥、lang chain') != '幂等性和密钥、LangGraph' or
                context.options(0)['hotwords'] != 'C++, .NET'):
            raise RuntimeError('Packaged remembered correction rule failed')
        state = load_rule_preferences(settings)
        before = settings.read_bytes()
        try:
            save_rule(settings, '密等性', '另一个词', state.fingerprint)
        except ValueError:
            pass
        else:
            raise RuntimeError('Packaged correction conflict silently overwritten')
        if settings.read_bytes() != before:
            raise RuntimeError('Packaged correction conflict changed existing rules')
        from qa_records import load_qa, replace_qa
        journal = Path(folder)/'interview.qa.jsonl'
        repeated = json.dumps(dict(question='为什么？', answer='相同回答'), ensure_ascii=False)+'\n'
        journal.write_text(repeated*2, encoding='utf-8')
        records = load_qa(journal)
        if records[0].key == records[1].key:
            raise RuntimeError('Packaged duplicate Q&A identities merged')
        replace_qa(records[0].origin, records[0].key, '第一题修订', '第一条新回答')
        revised = load_qa(journal)
        if [(item.question,item.answer) for item in revised] != [('第一题修订','第一条新回答'),('为什么？','相同回答')]:
            raise RuntimeError('Packaged Q&A revision changed another turn')
        original = journal.read_bytes()
        try:
            replace_qa(records[0].origin, records[0].key, '过期修订', '不应写入')
        except ValueError:
            pass
        else:
            raise RuntimeError('Packaged stale Q&A revision was not rejected')
        if journal.read_bytes() != original:
            raise RuntimeError('Packaged stale Q&A revision changed journal')
        from qa_records import save_qa_copy, append_qa
        copy = Path(folder)/'manual-copy.qa.jsonl'
        save_qa_copy(copy, [dict(id=item.key,time=item.time,question=item.question,answer=item.answer)
                            for item in revised])
        copied = load_qa(copy)
        if [(item.question,item.answer) for item in copied] != [(item.question,item.answer) for item in revised]:
            raise RuntimeError('Packaged manual Q&A copy changed content')
        append_qa(copy, 'new-followup', '再具体一点。', '使用订单号唯一约束。')
        if len(load_qa(copy)) != 3 or journal.read_bytes() != original:
            raise RuntimeError('Packaged manual Q&A continuation damaged its source')
    qa.set_enabled(True)
    qa.feed('什么是密等性？', record_key=0)
    qa.feed('什么是密等性？', record_key=1)
    qa.correct_transcript(0, '什么是幂等性？')
    if qa.pending['system'].parts != ['什么是幂等性？', '什么是密等性？']:
        raise RuntimeError('Packaged correction row identity failed')
    qa.set_enabled(False)
    qa.set_enabled(True)
    for technical_question in ('C++ 是什么？', 'C# 是什么？', 'C 是什么？'):
        qa.feed(technical_question)
        clock[0] += 2
        qa.tick()
        if not qa.active:
            raise RuntimeError('Packaged technical question identity collision')
        phases = [events.get(timeout=5)[1] for _ in range(3)]
        if phases[0][1:] != ('thinking', technical_question):
            raise RuntimeError('Packaged technical question changed')
        qa.active = False
    qa.set_enabled(False)
    assembler = WordAssembler()
    def technical_words(start, end, word):
        return [SimpleNamespace(words=[SimpleNamespace(start=start, end=end, word=word)])]
    first = assembler.consume(AudioChunk(np.zeros(15*RATE), 0, 0, forced=True),
                              technical_words(14., 14.5, 'C++'))
    second = assembler.consume(AudioChunk(np.zeros(2*RATE), 14.2, 14.2, overlap=True),
                               technical_words(.2, .8, ' C#'))
    if ''.join(row[2] for row in first+second) != 'C++ C#':
        raise RuntimeError('Packaged technical overlap word was lost')
    segmenter = PauseSegmenter(lambda audio: [dict(start=0, end=len(audio))])
    signal = np.arange(15*RATE, dtype=np.float32)
    first, = segmenter.push(signal, 0, 1000)
    tail, = segmenter.finish()
    if (not first.forced or not tail.overlap or tail.forced or
            not np.array_equal(tail.audio, signal[-int(2.4*RATE):]) or
            abs(tail.start - 12.6) > 1e-8 or abs(tail.epoch - 1012.6) > 1e-8 or
            segmenter.finish()):
        raise RuntimeError('Packaged short overlap context/time preservation failed')
    assembler = WordAssembler()
    before = assembler.consume(first, [SimpleNamespace(words=[
        SimpleNamespace(start=14., end=14.5, word='向量'),
        SimpleNamespace(start=14.6, end=15., word='数据库')])])
    after = assembler.consume(tail, [SimpleNamespace(words=[
        SimpleNamespace(start=1.4, end=1.9, word='向量'),
        SimpleNamespace(start=2., end=2.4, word='数据库')])])
    if ''.join(row[2] for row in before+after) != '向量数据库':
        raise RuntimeError('Packaged contextual tail duplicated or lost a technical word')
    from paragraphs import ParagraphAssembler
    idle_events = queue.Queue()
    engine = app.Transcriber(idle_events)
    def buffered_source():
        buffer = PauseSegmenter(lambda audio: [dict(start=0, end=len(audio))])
        buffer.push(np.ones(2*RATE, dtype=np.float32), 0, 1800000000.)
        return dict(received=-1., segmenter=buffer)
    states = dict(system=buffered_source(), microphone=buffered_source())
    paragraphs = ParagraphAssembler()
    def finish_source(source, chunks):
        for chunk in chunks:
            row = app.make_segment(chunk.start, chunk.start+len(chunk.audio)/RATE,
                                   '什么是幂等性？', chunk.epoch)
            row['source'] = source
            for completed in paragraphs.consume(row):
                engine.emit('segment', completed)
    engine._finish_idle_sources(states, finish_source, paragraphs,
        lambda source, active: engine.emit('voice_activity', dict(source=source, active=active)), 'microphone')
    completed, idle = idle_events.get_nowait(), idle_events.get_nowait()
    if (completed[0] != 'segment' or completed[1]['text'] != '什么是幂等性？' or
            completed[1]['source'] != 'system' or
            idle != ('voice_activity', dict(source='system', active=False)) or
            not idle_events.empty() or len(states['microphone']['segmenter'].audio) != 2*RATE):
        raise RuntimeError('Packaged independent idle-source finalization failed')
    def delayed_onsets(audio):
        edges = np.diff(np.pad((audio > 0).astype(int), (1, 1)))
        return [dict(start=int(a), end=int(b)) for a, b in
                zip(np.where(edges == 1)[0], np.where(edges == -1)[0]) if b-a > .16*RATE]
    signal = np.r_[np.ones(int(.5*RATE)), np.zeros(int(.58*RATE)),
                   np.ones(int(1.42*RATE))].astype(np.float32)
    segmenter = PauseSegmenter(delayed_onsets)
    ready = []
    for start in range(0,len(signal),RATE//10):
        ready.extend(segmenter.push(signal[start:start+RATE//10], start/RATE, 1800000000.+start/RATE))
    if ready or not segmenter.pending_speech or segmenter.confirmation_samples != round(.24*RATE):
        raise RuntimeError('Packaged pause confirmation split a new speech onset')
    final, = segmenter.finish()
    if not np.array_equal(final.audio, signal) or segmenter.pending_speech:
        raise RuntimeError('Packaged pause confirmation lost final speech')
    requests = []
    qa.runner = lambda prompt, cancel: requests.append(json.loads(prompt.split('\n', 1)[1])) or '本地排队验证'
    qa.set_enabled(True)
    view = SimpleNamespace(auto_qa=SimpleNamespace(get=lambda: True), rows=[], corrected_rows=set(),
                           qa=qa, qa_history=[], qa_image_contexts={})
    qa.voice_context_provider = lambda boundary: app.App.session_context_snapshot(view, source='auto', row_limit=boundary)
    qa.ask_clipboard('之前的问题')
    [events.get(timeout=5) for _ in range(3)]  # Keep busy until the UI accepts completion.
    for index, text in enumerate(('什么是密等性？', 'MySQL 慢查询怎么排查？')):
        view.rows.append(dict(text=text, source='system'))
        qa.feed(text, record_key=index)
        clock[0] += 2
        qa.tick()
    if len(qa.voice_queue) != 2:
        raise RuntimeError('Packaged finished voice turns merged while busy')
    view.rows[0]['text'] = '什么是幂等性？'
    view.corrected_rows.add(0)
    qa.correct_transcript(0, '什么是幂等性？')
    view.qa_history = [('之前的问题', '上一个完整回答')]
    qa.active = False
    qa.tick()
    phases = [events.get(timeout=5)[1] for _ in range(3)]
    background = json.dumps(requests[-1]['会话资料'], ensure_ascii=False)
    if (phases[0][2] != '什么是幂等性？' or 'MySQL' in background or
            '密等性' in background or '上一个完整回答' not in background):
        raise RuntimeError('Packaged queued question correction/context boundary failed')
    qa.active = False
    qa.tick()
    phases = [events.get(timeout=5)[1] for _ in range(3)]
    if phases[0][2] != 'MySQL 慢查询怎么排查？':
        raise RuntimeError('Packaged queued voice order failed')
    qa.set_enabled(False)
    boundary = RecognitionContext('net=Network, .NET=DotNet')
    if boundary.normalize('file.NET net.io net.') != 'file.NET net.io Network.':
        raise RuntimeError('Packaged explicit alias changed qualified technical names')
    qa.set_enabled(True)
    qa.context_provider = lambda: ([], [('什么是幂等性？', '相同订单只处理一次。')])
    qa.feed('再具体一点。')
    clock[0] += 2
    qa.tick()
    phases = [events.get(timeout=5)[1] for _ in range(3)]
    if phases[0][2] != '再具体一点。':
        raise RuntimeError('Packaged restored voice follow-up was lost')
    qa.set_enabled(False)
    qa.context_provider = None
    qa.set_enabled(True)
    qa.feed('介绍一下幂等性。')
    clock[0] += 2
    qa.collect_voice()
    qa.feed('再具体一点。')
    clock[0] += 2
    qa.collect_voice()
    if [turn.question.text() for turn in qa.voice_queue] != ['介绍一下幂等性。','再具体一点。']:
        raise RuntimeError('Packaged waiting voice follow-up was lost')
    qa.set_enabled(False)
    return dict(terminology=canonical, presets=list(INTERVIEW_HOTWORDS),
                **verify_recording_io(app),
                stereo_protection=True, complete_question_wait=True,
                interviewer_source_routing=True, recording_append=True,
                spoken_commands=True, contextual_followups=True, recent_duplicate_suppression=True,
                complete_hotword_budget=True, long_alias_preservation=True,
                transcript_correction_overlay=True, correction_row_identity=True,
                technical_question_identity=True, technical_overlap_preservation=True,
                independent_voice_queue=True, queued_transcript_boundary=True, queued_correction=True,
                remembered_correction_rules=True, qualified_alias_boundaries=True,
                correction_prompt_isolation=True, qa_journal_identity=True,
                qa_journal_conflict_guard=True, historical_voice_followups=True,
                manual_qa_copy=True, short_tail_context=True, idle_source_finalization=True,
                terminal_pause_confirmation=True)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    if '--self-test' in sys.argv:
        report = Path(sys.argv[sys.argv.index('--self-test') + 1])
        try:
            if '--verify-bundled-model' in sys.argv:
                import os
                os.environ['HF_HUB_OFFLINE'] = '1'
            import app
            import tkinter as tk
            import numpy as np
            from faster_whisper.vad import get_speech_timestamps, VadOptions
            from updates import version_tuple
            from app_paths import RESOURCES
            root = tk.Tk()
            root.withdraw()
            image = tk.PhotoImage(file=str(RESOURCES / 'assets/logo-24.png'))
            get_speech_timestamps(np.zeros(16000, dtype=np.float32), VadOptions())
            from echo_cancellation import make_processor
            echo = make_processor()
            echo_audio = echo.process(np.zeros(160, np.float32), np.zeros(160, np.float32))
            if echo_audio.shape != (160,) or not np.isfinite(echo_audio).all():
                raise RuntimeError('Packaged echo cancellation self-test failed')
            root.destroy()
            interview_runtime = verify_interview_runtime(app)
            interview_runtime.update(verify_segmentation_policy())
            interview_runtime.update(verify_live_corrections())
            interview_runtime.update(verify_saved_wordlists())
            interview_runtime.update(verify_pending_words())
            interview_runtime.update(verify_incomplete_voice())
            interview_runtime.update(verify_capture_loss_tails())
            interview_runtime.update(verify_append_recovery(app))
            interview_runtime.update(verify_storage_stop_qa(app))
            knowledge_package_sources = []
            delivery_verified = False
            if '--verify-bundled-delivery' in sys.argv:
                from bundled_delivery import bootstrap_delivery
                from reference_files import load_settings
                bootstrap_delivery()
                delivery_settings = load_settings()
                if not delivery_settings['enabled'] or not delivery_settings['paths']:
                    raise RuntimeError('Bundled delivery did not enable customer knowledge')
                delivery_verified = True
            if '--verify-knowledge-package' in sys.argv:
                import threading
                from knowledge_packages import KnowledgeLibrary
                package_path = sys.argv[sys.argv.index('--verify-knowledge-package') + 1]
                package_result = json.loads(KnowledgeLibrary().context(
                    {'enabled': True, 'paths': [package_path]},
                    '数据库已更新，但通知没有发出去怎么办？', [], threading.Event()))
                knowledge_package_sources = [item['章节'] for item in package_result['片段']]
                if not knowledge_package_sources:
                    raise RuntimeError('Packaged knowledge retrieval returned no sources')
            if '--verify-bundled-model' in sys.argv:
                from model_download import cached_model, BUNDLED_MODELS
                from knowledge_embedding import Encoder, BUNDLED_MODEL_DIR
                knowledge_encoder = Encoder(BUNDLED_MODEL_DIR)
                knowledge_vector = knowledge_encoder.encode(['知识包检索验证'], query=True)
                if knowledge_vector.shape != (1, 512) or not np.isfinite(knowledge_vector).all():
                    raise RuntimeError('Bundled knowledge model failed')
                del knowledge_encoder
                from model_runtime import load_model
                if cached_model('small') != BUNDLED_MODELS / 'small':
                    raise RuntimeError('Bundled model missing; refusing cache fallback')
                model = load_model('small', device='cpu')
                del model
            if '--verify-download-worker' in sys.argv:
                from model_download import download_worker
                context = multiprocessing.get_context('spawn')
                receiver, sender = context.Pipe(duplex=False)
                worker = context.Process(target=download_worker, args=('small', sender), daemon=True)
                worker.start()
                sender.close()
                try:
                    while receiver.poll(60):
                        kind, message = receiver.recv()
                        if kind == 'error':
                            raise RuntimeError(message)
                        if kind == 'done':
                            break
                    else:
                        raise TimeoutError('Download worker did not finish')
                finally:
                    worker.join(2)
                    if worker.is_alive():
                        worker.terminate()
                        worker.join(2)
                    receiver.close()
            if '--verify-model' in sys.argv:
                from model_runtime import load_model
                model = load_model(sys.argv[sys.argv.index('--verify-model') + 1], device='cpu')
                segments, _ = model.transcribe(np.zeros(16000, dtype=np.float32), language='zh', beam_size=1, max_new_tokens=1)
                list(segments)
            if '--verify-auto-models' in sys.argv:
                from model_runtime import load_model
                for name in ('small', 'large-v3-turbo', 'small'):
                    model = load_model(name)
                    del model
            gpu_verified = False
            if '--verify-gpu' in sys.argv:
                from model_runtime import load_model
                model = load_model('small', device='cuda')
                if model.model.device != 'cuda':
                    raise RuntimeError('GPU verification unexpectedly fell back to CPU')
                del model
                gpu_verified = True
            qwen_verified = None
            if '--verify-qwen-engine' in sys.argv:
                audio_path = (sys.argv[sys.argv.index('--verify-qwen-audio')+1]
                              if '--verify-qwen-audio' in sys.argv else None)
                qwen_verified = verify_optional_qwen(audio_path)
            capture_verified = None
            if '--verify-capture-model' in sys.argv:
                capture_verified = verify_capture_owner('cuda' if '--verify-gpu' in sys.argv else 'auto')
            report.write_text(json.dumps({'ok': True, 'version': app.VERSION, 'echo_verified': True,
                                         'interview_runtime': interview_runtime,
                                         'bundled_delivery_verified': delivery_verified,
                                         'knowledge_model_verified': '--verify-bundled-model' in sys.argv,
                                         'knowledge_package_sources': knowledge_package_sources,
                                         'gpu_verified': gpu_verified,
                                         'qwen_engine': qwen_verified,
                                         'capture_model': capture_verified,
                                         'bundled_model': 'small' if '--verify-bundled-model' in sys.argv else None}), encoding='utf-8')
        except Exception:
            import traceback
            report.write_text(traceback.format_exc(), encoding='utf-8')
            raise
    else:
        import runpy
        # A real import keeps app visible to the freezer's dependency analysis.
        import app
        runpy.run_module('app', run_name='__main__')
