#!/usr/bin/env python3
"""Local sequential ONNX ASR comparison. Audio never leaves this machine."""
from __future__ import annotations
import argparse
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
MODELS = {'parakeet': 'nemo-parakeet-tdt-0.6b-v3', 'gigaam': 'gigaam-v3-e2e-rnnt', 'gigaam-ctc': 'gigaam-v3-e2e-ctc'}


def save(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def normalized(text):
    text = unicodedata.normalize('NFKC', text).lower().replace('ё', 'е')
    return ' '.join(''.join(c if c.isalnum() else ' ' for c in text).split())


def prepare(args):
    import numpy as np
    import onnx_asr
    import soundfile as sf
    import onnxruntime as ort
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    wav = out / 'input.wav'
    cmd = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(args.audio)]
    if args.seconds:
        cmd += ['-t', str(args.seconds)]
    subprocess.run(cmd + ['-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(wav)], check=True)
    audio, sr = sf.read(wav, dtype='float32')
    if not len(audio):
        raise ValueError('Empty audio')
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    vad = onnx_asr.load_vad('silero', providers=['CPUExecutionProvider'], sess_options=options)
    t = time.perf_counter()
    groups = vad.segment_batch(audio[None, :], np.array([len(audio)], dtype=np.int64), sr,
        min_silence_duration_ms=500, max_speech_duration_s=args.chunk_seconds,
        min_speech_duration_ms=250, speech_pad_ms=100)
    segments = [{'start': int(s), 'end': int(e)} for group in groups for s, e in group]
    if not segments:
        raise ValueError('VAD found no speech')
    assert all(0 <= x['start'] < x['end'] <= len(audio) and (x['end']-x['start'])/sr <= args.chunk_seconds + .01 for x in segments)
    manifest = {'source': str(args.audio.resolve()), 'audio_sha256': hashlib.sha256(wav.read_bytes()).hexdigest(),
        'sample_rate': sr, 'duration_s': len(audio)/sr, 'segments': segments,
        'speech_s': sum(x['end']-x['start'] for x in segments)/sr,
        'vad_s': time.perf_counter()-t, 'chunk_seconds': args.chunk_seconds,
        'threads': args.threads, 'quantization': args.quantization,
        'platform': platform.platform(), 'machine': platform.machine(),
        'versions': {n: importlib.metadata.version(n) for n in ['onnx-asr', 'onnxruntime', 'numpy']}}
    save(out / 'manifest.json', manifest)
    print(f"Prepared {manifest['duration_s']:.1f}s, {len(segments)} shared segments, {manifest['speech_s']:.1f}s speech", flush=True)


def worker(args):
    import numpy as np
    import onnx_asr
    import onnxruntime as ort
    import psutil
    import soundfile as sf
    import threading
    out = args.output
    manifest = json.loads((out / 'manifest.json').read_text())
    audio, sr = sf.read(out / 'input.wav', dtype='float32')
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = manifest['threads']
    opts.inter_op_num_threads = 1
    opts.enable_mem_pattern = False
    t = time.perf_counter()
    model = onnx_asr.load_model(MODELS[args.worker], quantization=manifest['quantization'] if manifest['quantization'] != 'fp32' else None,
        providers=['CPUExecutionProvider'], sess_options=opts)
    load_s = time.perf_counter()-t
    first = manifest['segments'][0]
    t = time.perf_counter()
    model.recognize(audio[first['start']:first['end']], sample_rate=sr)
    warmup_s = time.perf_counter()-t
    proc = psutil.Process()
    peak = proc.memory_info().rss
    stop = threading.Event()
    def sample():
        nonlocal peak
        while not stop.wait(.05):
            peak = max(peak, proc.memory_info().rss)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    rows = []
    begun = time.perf_counter()
    try:
        for i, seg in enumerate(manifest['segments']):
            t = time.perf_counter()
            text = model.recognize(audio[seg['start']:seg['end']], sample_rate=sr)
            elapsed = time.perf_counter()-t
            rows.append({'start_s': seg['start']/sr, 'end_s': seg['end']/sr, 'text': str(text), 'inference_s': elapsed})
            if i % 20 == 0:
                print(f"{args.worker}: {i+1}/{len(manifest['segments'])}", flush=True)
                save(out / f'{args.worker}.partial.json', rows)
    finally:
        stop.set()
        thread.join()
    wall = time.perf_counter()-begun
    infer = sum(row['inference_s'] for row in rows)
    durations = np.array([row['end_s']-row['start_s'] for row in rows])
    times = np.array([row['inference_s'] for row in rows])
    result = {'model': MODELS[args.worker], 'quantization': manifest['quantization'], 'provider': 'CPUExecutionProvider',
        'load_and_download_s': load_s, 'warmup_s': warmup_s, 'inference_s': infer, 'loop_wall_s': wall,
        'rtf_speech': infer/manifest['speech_s'], 'rtf_recording': infer/manifest['duration_s'],
        'rtfx_speech': manifest['speech_s']/infer, 'p50_chunk_s': float(np.median(times)),
        'p95_chunk_s': float(np.percentile(times, 95)), 'chunks_slower_than_realtime': int((times > durations).sum()),
        'peak_rss_mib': peak/1024**2, 'segments': rows}
    if args.reference:
        from jiwer import wer, cer
        ref = normalized(args.reference.read_text(encoding='utf-8'))
        hyp = normalized(' '.join(r['text'] for r in rows))
        if not ref:
            raise ValueError('Reference is empty')
        result.update(wer=wer(ref, hyp), cer=cer(ref, hyp))
    save(out / f'{args.worker}.json', result)
    (out / f'{args.worker}.txt').write_text('\n'.join(f"[{r['start_s']:.2f}–{r['end_s']:.2f}] {r['text']}" for r in rows), encoding='utf-8')
    print(f"{args.worker}: RTF={result['rtf_speech']:.3f}, {result['rtfx_speech']:.1f}x realtime, peak RSS={result['peak_rss_mib']:.0f}MiB", flush=True)


def report(out, names):
    data = {name: json.loads((out / f'{name}.json').read_text()) for name in names}
    def esc(x): return html.escape(str(x))
    heads = ''.join(f"<th>{esc(n)}<br><small>{esc(data[n]['model'])}</small></th>" for n in names)
    metrics = [('Распознавание, с', 'inference_s'), ('Быстрее реального времени, × (по речи)', 'rtfx_speech'),
        ('RTF по речи (меньше лучше)', 'rtf_speech'), ('p95 обработки фрагмента, с', 'p95_chunk_s'),
        ('Фрагментов медленнее реального времени', 'chunks_slower_than_realtime'), ('Пиковая RSS после загрузки, МиБ', 'peak_rss_mib')]
    if all('wer' in d for d in data.values()): metrics += [('WER', 'wer'), ('CER', 'cer')]
    summary = ''.join('<tr><td>'+label+'</td>'+''.join(f'<td>{data[n][key]:.3f}</td>' for n in names)+'</tr>' for label,key in metrics)
    rows = []
    for i, segment in enumerate(data[names[0]]['segments']):
        start, end = segment['start_s'], segment['end_s']
        cells = ''.join(f"<td>{esc(data[n]['segments'][i]['text'])}<small>{data[n]['segments'][i]['inference_s']:.2f} с</small></td>" for n in names)
        rows.append(f'<tr><td><button onclick="play({start},{end})">▶ {int(start//60):02}:{int(start%60):02}</button></td>{cells}</tr>')
    page = f'''<!doctype html><html lang="ru"><meta charset="utf-8"><title>GigaAM / Parakeet</title>
<style>body{{font:16px system-ui;margin:32px;color:#182334;background:#f6f8fa}}table{{border-collapse:collapse;width:100%;background:white;margin:24px 0}}td,th{{border:1px solid #d8dfe6;padding:12px;text-align:left;vertical-align:top}}th{{background:#e7eef5;position:sticky;top:75px}}small{{display:block;color:#687585;margin-top:6px}}audio{{width:min(900px,100%)}}.player{{position:sticky;top:0;background:#f6f8fa;padding:10px;z-index:2}}button{{cursor:pointer;white-space:nowrap}}td:first-child{{width:95px}}</style>
<h1>GigaAM / Parakeet — локальное сравнение</h1>
<p>CPU, точность согласно manifest.json, одинаковые фрагменты Silero VAD. Модели запускаются последовательно в отдельных процессах после прогрева. Время скачивания/загрузки, VAD и декодирования файла не входит во время ASR.</p>
<p>Это Python/ONNX-прототип, а не замер Meetily. Без ручного эталона WER неизвестен; различие текстов не доказывает, какая модель точнее. Метки относятся к фрагментам. RSS включает Python и аудиобуфер, измерена после загрузки и не отражает пик скачивания/загрузки.</p>
<div class="player"><audio id="audio" controls src="input.wav"></audio></div>
<table><tr><th>Метрика</th>{heads}</tr>{summary}</table>
<table><tr><th>Аудио</th>{heads}</tr>{''.join(rows)}</table>
<script>const a=document.getElementById('audio');let end=Infinity;function play(s,e){{end=e;a.currentTime=s;a.play()}}a.addEventListener('timeupdate',()=>{{if(a.currentTime>=end){{a.pause();end=Infinity}}}});</script></html>'''
    (out / 'report.html').write_text(page, encoding='utf-8')
    save(out / 'summary.json', {n:{k:v for k,v in d.items() if k!='segments'} for n,d in data.items()})
    print(f"Report: {out / 'report.html'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('audio', type=Path, nargs='?', default=ROOT/'audio/meeting.m4a')
    p.add_argument('--output', type=Path, default=ROOT/'results/full')
    p.add_argument('--models', nargs='+', choices=MODELS, default=['parakeet', 'gigaam'])
    p.add_argument('--seconds', type=float, help='Compare first N seconds only')
    p.add_argument('--chunk-seconds', type=float, default=20)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--quantization', choices=['int8', 'fp32'], default='int8')
    p.add_argument('--reference', type=Path, help='Manually corrected plain text for the exact compared audio')
    p.add_argument('--worker', choices=MODELS, help=argparse.SUPPRESS)
    p.add_argument('--report-only', action='store_true')
    args = p.parse_args()
    args.output = args.output.resolve()
    if args.worker:
        worker(args)
    elif args.report_only:
        report(args.output, args.models)
    else:
        if not 1 <= args.chunk_seconds <= 24 or args.threads < 1 or (args.seconds is not None and args.seconds <= 0):
            p.error('chunk-seconds must be 1..24, threads >= 1, seconds > 0')
        if (args.output / 'manifest.json').exists():
            p.error('Output already contains a run; choose another --output directory')
        prepare(args)
        for name in args.models:
            cmd = [sys.executable, str(Path(__file__).resolve()), '--worker', name, '--output', str(args.output)]
            if args.reference: cmd += ['--reference', str(args.reference.resolve())]
            subprocess.run(cmd, check=True)
        report(args.output, args.models)

if __name__ == '__main__':
    main()
