"""Compare append-only saving with the former whole-TXT rewrite on local disk.

This measures formatting and file writes, not total UI or ASR latency. All data
is synthetic and temporary; no capture, playback, API calls or settings changes.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import export_text, save_segment


def legacy_save(path, row, rows):
    with path.open('a', encoding='utf-8') as file:
        file.write(json.dumps(row, ensure_ascii=False)+'\n')
    path.with_suffix('.txt').write_text(export_text(rows), encoding='utf-8-sig')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--rows', nargs='+', type=int, default=[0, 300, 1200, 3000])
    parser.add_argument('--new-rows', type=int, default=30)
    args = parser.parse_args()
    if min(args.rows) < 0 or args.new_rows < 1:
        parser.error('row counts must be nonnegative; --new-rows must be positive')
    report = {'scope': 'Synthetic local formatting/file-write cost only', 'cases': []}
    with tempfile.TemporaryDirectory(prefix='wenlu-save-benchmark-') as folder:
        for count in args.rows:
            seed = [dict(start=i*3, end=i*3+2, text='我们讨论消息队列的幂等性与失败恢复方案。'*3,
                         captured_at='2026-11-02T10:00:00+08:00') for i in range(count)]
            outputs, timings = {}, {}
            for mode in ('rewrite', 'append'):
                path = Path(folder)/(mode+'.jsonl')
                path.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in seed), encoding='utf-8')
                txt = path.with_suffix('.txt')
                if seed:
                    txt.write_text(export_text(seed), encoding='utf-8-sig')
                else:
                    txt.unlink(missing_ok=True)
                rows, elapsed = list(seed), []
                for index in range(args.new_rows):
                    row = dict(start=count*3+index, end=count*3+index+1,
                               text='这是面试过程中新增加的一段回答。', captured_at='2026-11-02T10:00:00+08:00')
                    rows.append(row)
                    started = time.perf_counter()
                    if mode == 'rewrite':
                        legacy_save(path, row, rows)
                    else:
                        save_segment(path, row)
                    elapsed.append((time.perf_counter()-started)*1000)
                timings[mode] = dict(mean_ms=statistics.mean(elapsed), median_ms=statistics.median(elapsed),
                                     p95_ms=sorted(elapsed)[min(len(elapsed)-1, int(len(elapsed)*.95))])
                outputs[mode] = path.read_bytes(), txt.read_bytes()
            assert outputs['rewrite'] == outputs['append'], 'Recording contents changed'
            result = dict(existing_rows=count, new_rows=args.new_rows, timings=timings,
                          identical_recording_bytes=True,
                          speedup=timings['rewrite']['mean_ms']/max(1e-9, timings['append']['mean_ms']))
            report['cases'].append(result)
            print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
