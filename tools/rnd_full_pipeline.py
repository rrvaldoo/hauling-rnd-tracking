"""Colab experiment harness; production hauling modules remain unchanged."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import csv
from dataclasses import replace
import gc
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import types
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def profiled_pipeline_module():
    """Add timing hooks to an in-memory copy, failing if the source contract changes."""
    import hauling.engine.pipeline as original
    source = inspect.getsource(original)
    edits = [
        ('def run_pipeline(options: RunOptions, adapter_factory=YoloAdapter) -> dict:',
         'def run_pipeline(options: RunOptions, adapter_factory=YoloAdapter, profiler=None) -> dict:'),
        ('capture = cv2.VideoCapture(str(options.video.resolve()))',
         'capture = (profiler.capture(cv2, str(options.video.resolve())) if profiler else cv2.VideoCapture(str(options.video.resolve())))'),
        ('            while ok:\n', '            while ok:\n                if profiler: profiler.begin_frame()\n'),
        ('                if summary["total_frames_read"] % 250 == 0:',
         '                if profiler: profiler.end_frame(index)\n                if summary["total_frames_read"] % 250 == 0:'),
    ]
    for old, new in edits:
        if source.count(old) != 1:
            raise RuntimeError('Pipeline timing hook no longer matches snapshot')
        source = source.replace(old, new, 1)
    name = '_hauling_rnd_profiled_pipeline'
    module = types.ModuleType(name)
    module.__file__ = original.__file__
    sys.modules[name] = module
    exec(compile(source, original.__file__ + '[timing hooks]', 'exec'), module.__dict__)
    return module


class Profiler:
    def __init__(self, cuda=True):
        self.cuda = cuda
        self.samples = defaultdict(list)
        self.frames = []
        self.pending_decode = 0.0

    def sync(self):
        if self.cuda:
            import torch
            torch.cuda.synchronize()

    @contextmanager
    def measure(self, stage):
        self.sync()
        started = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            self.samples[stage].append(1000 * (time.perf_counter() - started))

    def capture(self, cv2, path):
        owner = self
        cap = cv2.VideoCapture(path)

        class Capture:
            def __getattr__(self, name):
                return getattr(cap, name)

            def read(self):
                started = time.perf_counter()
                result = cap.read()
                owner.pending_decode = 1000 * (time.perf_counter() - started)
                return result
        return Capture()

    def begin_frame(self):
        self.sync()
        self.frame_start = time.perf_counter()

    def end_frame(self, index):
        self.sync()
        body = 1000 * (time.perf_counter() - self.frame_start)
        self.frames.append(dict(frame_index=index, decode_ms=self.pending_decode,
                                pipeline_frame_ms=body + self.pending_decode))

    def save(self, directory):
        import numpy as np
        with (directory / 'frame_latency.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['frame_index', 'decode_ms', 'pipeline_frame_ms'])
            writer.writeheader()
            writer.writerows(self.frames)
        stages = {}
        for name, data in self.samples.items():
            stages[name] = dict(calls=len(data), total_ms=sum(data),
                                median_ms=float(np.median(data)), p95_ms=float(np.percentile(data, 95)))
        (directory / 'stage_latency.json').write_text(json.dumps(stages, indent=2))
        warm = self.frames[30:]
        values = [r['pipeline_frame_ms'] for r in warm]
        return dict(frame_warmup_excluded=min(30, len(self.frames)),
                    frame_ms_median=float(np.median(values)) if values else None,
                    frame_ms_p95=float(np.percentile(values, 95)) if values else None,
                    active_frame_fps=1000 * len(values) / sum(values) if values else None,
                    timing_note='Frame metrics include decode, OCR, business logic, trace, report/crop IO and render/write; exclude startup and final drain. Nested stage timings must not be summed.')


def deepsort_assignments(tracker, boxes, frame):
    """Map only confirmed tracks observed in this frame to original detector indices."""
    import numpy as np
    raw, others = [], []
    for i, (box, score, cls) in enumerate(zip(boxes.xyxy, boxes.conf, boxes.cls)):
        x1, y1, x2, y2 = map(float, box)
        if x2 <= x1 or y2 <= y1:
            continue
        raw.append(([x1, y1, x2-x1, y2-y1], float(score), int(cls)))
        others.append({'detection_index': i})
    tracks = tracker.update_tracks(raw, frame=frame, others=others)
    assigned = {}
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update != 0:
            continue
        info = track.get_det_supplementary()
        if info is None:
            continue
        idx = info['detection_index']
        if idx in assigned or not 0 <= idx < len(boxes):
            raise RuntimeError('Invalid DeepSORT detection mapping')
        box = np.asarray(track.to_ltrb(), dtype=float)
        if not np.isfinite(box).all():
            raise RuntimeError('Non-finite DeepSORT box')
        assigned[idx] = [*box, int(track.track_id), float(boxes.conf[idx]),
                         int(boxes.cls[idx]), idx]
    return assigned


def adapter_factory(spec, profiler, engine=None):
    from hauling.vision.detection import YoloAdapter, Detection

    class ExperimentAdapter(YoloAdapter):
        def __init__(self, *args):
            from ultralytics import YOLO
            from ultralytics.trackers.bot_sort import BOTSORT
            with profiler.measure('adapter_initialization'):
                super().__init__(*args)
                fps = args[3]
                if spec == 'botsort':
                    settings = dict(vars(self.tracker.args), gmc_method='sparseOptFlow',
                                    proximity_thresh=.5, appearance_thresh=.8,
                                    with_reid=False, model='auto')
                    self.tracker = BOTSORT(types.SimpleNamespace(**settings), frame_rate=fps)
                elif spec == 'deepsort':
                    from deep_sort_realtime.deepsort_tracker import DeepSort
                    self.tracker = DeepSort(max_age=max(1, int(fps * args[6])), n_init=3,
                        max_iou_distance=.7, max_cosine_distance=.2, nn_budget=100,
                        nms_max_overlap=1.0, embedder='mobilenet', embedder_gpu=self.device.startswith('cuda'),
                        half=False, bgr=True)
                elif spec == 'tensorrt_fp16':
                    if engine is None or not Path(engine).is_file():
                        raise ValueError('TensorRT experiment requires a local engine')
                    self.vehicle = YOLO(str(engine), task='detect')

        def detect(self, frame):
            cfg = self.config
            half = spec in {'pytorch_fp16', 'tensorrt_fp16'}
            with profiler.measure('vehicle_detector_with_prepost_transfer'):
                result = self.vehicle.predict(frame, conf=cfg.confidence, iou=cfg.iou,
                    imgsz=cfg.image_size, rect=False, max_det=cfg.max_detections,
                    agnostic_nms=cfg.agnostic_nms, device=self.vehicle_device,
                    verbose=False, save=False, half=half)[0]
                boxes = result.boxes.cpu().numpy()
                if spec == 'tensorrt_fp16' and not self.vehicle.predictor.model.fp16:
                    raise RuntimeError('TensorRT engine is not FP16')
            with profiler.measure('tracker_including_reid_or_gmc'):
                if spec == 'deepsort':
                    assigned = deepsort_assignments(self.tracker, boxes, frame)
                else:
                    assigned = {int(row[-1]): row for row in self.tracker.update(boxes, frame)}
            output = []
            for i, (box, confidence, cls) in enumerate(zip(boxes.xyxy, boxes.conf, boxes.cls)):
                row = assigned.get(i)
                output.append(Detection(
                    tuple(float(v) for v in (row[:4] if row is not None else box)),
                    float(row[5]) if row is not None else float(confidence),
                    int(row[4]) if row is not None else None,
                    tuple(float(v) for v in box), float(confidence), int(cls),
                    str(self.vehicle.names[int(cls)])))
            return output, len(boxes)

        def detect_plate(self, *args, **kwargs):
            with profiler.measure('hull_detector_and_reference'):
                return super().detect_plate(*args, **kwargs)
    return ExperimentAdapter


def run_experiment(root, output, spec, engine=None, max_seconds=45, device='cuda:0'):
    import cv2
    import torch
    import hauling.engine.pipeline as production
    from hauling.vision.recognition import CharacterModels, HullRecognizer
    from hauling.vision.vehicle_type import VehicleTypeResolver
    root, output = Path(root), Path(output)
    if output.exists():
        raise ValueError('Use a fresh output directory to preserve earlier results')
    pipeline = profiled_pipeline_module()
    profiler = Profiler(cuda=device.startswith('cuda'))
    models = root / 'model-requirement'
    paths = CharacterModels(
        models/'character_angka_detection_hauling_yolov8m_V2.pt',
        models/'character_huruf_detection_hauling_yolov8m_V2.pt',
        models/'classification_hauling_angka_yolo8m_imgz96_v9.pt',
        models/'classification_hauling_huruf_yolo8m_imgz96_v9.pt')
    options = pipeline.RunOptions(
        video=root/'video/20261007133257-3201.mp4',
        vehicle_model=models/'Hauling_Vehicle_yolov8m_SingleCls_V11.pt',
        plate_model=models/'No_Lambung_TruckHD_Detection_yolov8m_V3.pt',
        config=models/'camera_offline.conf', output_dir=output, save_video=True,
        report=True, save_hull_crops=True, log_detections=True,
        max_seconds=max_seconds, device=device, tracker_buffer_seconds=1.,
        tracker_config=Path(production.__file__).resolve().parents[1]/'bytetrack.yaml',
        hull_reference=root/'reference/panel_closeup.jpg', character_models=paths,
        vehicle_classifier=models/'Classification_truck_mitra_yolov8m_Imgz224_V3.pt',
        character_confidence=.5, ocr_sample_seconds=.2, ocr_lookahead_seconds=.3)
    # Timing wrappers preserve each function's return value and exception behavior.
    original_ocr, original_type = HullRecognizer.recognize, VehicleTypeResolver.resolve
    def recognize(instance, *args, **kwargs):
        with profiler.measure('ocr_full_read'):
            return original_ocr(instance, *args, **kwargs)
    def resolve(instance, *args, **kwargs):
        with profiler.measure('vehicle_type_resolve'):
            return original_type(instance, *args, **kwargs)
    cv2.setNumThreads(1)
    profiler.sync()
    started = time.perf_counter()
    with patch.object(HullRecognizer, 'recognize', recognize), patch.object(VehicleTypeResolver, 'resolve', resolve):
        summary = pipeline.run_pipeline(options, adapter_factory(spec, profiler, engine), profiler=profiler)
    profiler.sync()
    wall = time.perf_counter() - started
    performance = dict(variant=spec, cold_full_pipeline_seconds=wall,
        cold_full_pipeline_fps=summary['total_frames_read']/wall,
        runtime_device=device, vehicle_precision='FP16' if 'fp16' in spec else 'FP32',
        auxiliary_models=f'Original .pt FP32 on {device}; includes hull, OCR and vehicle classifier',
        detector_rect=False, detector_shape=[640,640],
        deepsort_embedding='MobileNetV2 ImageNet bottleneck FP32' if spec=='deepsort' else None,
        botsort_reid=False if spec=='botsort' else None,
        **profiler.save(output))
    if profiler.cuda:
        performance['torch_peak_allocated_mb'] = torch.cuda.max_memory_allocated()/1024**2
        performance['torch_memory_note'] = 'PyTorch allocator only; does not measure total TensorRT/CUDA VRAM.'
    performance['warnings'] = summary['warnings']
    performance['acceptance_issues'] = [key for key in ('plate_inference_errors','character_recognition_errors',
        'vehicle_type_refinement_errors','report_errors') if summary.get(key,0)]
    if not summary['video_saved'] or not summary['report_saved']:
        performance['acceptance_issues'].append('missing_video_or_report')
    expected = math.ceil(max_seconds*summary['fps'])
    if summary['total_frames_read'] != expected:
        performance['acceptance_issues'].append('frame_count_mismatch')
    (output/'performance.json').write_text(json.dumps(performance, indent=2))
    print(json.dumps(performance, indent=2), flush=True)
    return performance


def compare_event_rows(reference, candidate, tolerance=.5):
    """Timestamp/direction agreement to baseline, not accuracy or ID ground truth."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    if not reference or not candidate:
        return [], list(range(len(reference))), list(range(len(candidate)))
    costs = np.full((len(reference),len(candidate)), 1e6)
    for i,a in enumerate(reference):
        for j,b in enumerate(candidate):
            delta = abs(float(a['event_timestamp_s'])-float(b['event_timestamp_s']))
            if a['travel_direction']==b['travel_direction'] and delta<=tolerance:
                costs[i,j]=delta
    ii,jj=linear_sum_assignment(costs)
    pairs=[(int(i),int(j)) for i,j in zip(ii,jj) if costs[i,j]<1e6]
    return pairs,[i for i in range(len(reference)) if i not in {p[0] for p in pairs}], [j for j in range(len(candidate)) if j not in {p[1] for p in pairs}]


def summarize(session):
    import pandas as pd
    rows=[]
    session=Path(session)
    for path in sorted(session.glob('*/run-*/performance.json')):
        perf=json.loads(path.read_text())
        summary=json.loads(path.with_name('run_summary.json').read_text())
        row={k:v for k,v in perf.items() if isinstance(v,(str,int,float)) or v is None}
        row.update({k:summary.get(k) for k in ['events_csv','speed_valid','overspeed','unique_tracks',
            'hull_text_recognized','hull_text_complete','plate_detected','total_frames_read',
            'plate_inference_errors','character_recognition_errors','vehicle_type_refinement_errors']})
        row['run']=path.parent.name
        row['valid_run']=not perf['acceptance_issues'] and not summary['warnings']
        row['acceptance_issues']=';'.join(perf['acceptance_issues'])
        rows.append(row)
    df=pd.DataFrame(rows)
    df.to_csv(session/'comparison_runs.csv',index=False)
    if len(df):
        valid=df[df['valid_run']]
        if len(valid):
            valid.groupby('variant').median(numeric_only=True).to_csv(session/'comparison_median.csv')
    agreements=[]
    with (session/'bytetrack/run-01/events.csv').open(newline='') as f:
        baseline=list(csv.DictReader(f))
    for path in sorted(session.glob('*/run-01/events.csv')):
        with path.open(newline='') as f:candidate=list(csv.DictReader(f))
        pairs,missing,extra=compare_event_rows(baseline,candidate)
        details=[]
        for i,j in pairs:
            a,b=baseline[i],candidate[j]
            bothspeed=bool(a['speed_kmh'] and b['speed_kmh'])
            details.append(dict(baseline_event=a['event_id'],candidate_event=b['event_id'],
                event_time_delta_s=float(b['event_timestamp_s'])-float(a['event_timestamp_s']),
                speed_delta_kmh=float(b['speed_kmh'])-float(a['speed_kmh']) if bothspeed else None,
                baseline_text=a['plate_text'],candidate_text=b['plate_text'],
                nonempty_text_equal=bool(a['plate_text'] and b['plate_text'] and a['plate_text']==b['plate_text']),
                violation_equal=a['violation_status']==b['violation_status']))
        (path.parent/'agreement_to_bytetrack.json').write_text(json.dumps(dict(
            label='Agreement to baseline; not ground-truth accuracy', matching='Hungarian timestamp <= 0.5 s and same direction; ambiguous near-simultaneous vehicles need manual review',
            matched=details,unmatched_baseline_event_ids=[baseline[i]['event_id'] for i in missing],
            unmatched_candidate_event_ids=[candidate[j]['event_id'] for j in extra]),indent=2))
        agreements.append(dict(variant=path.parent.parent.name,matched_events=len(pairs),
            unmatched_baseline=len(missing),unmatched_candidate=len(extra),
            matching_rule='timestamp<=0.5s, same direction; not ground truth'))
    pd.DataFrame(agreements).to_csv(session/'event_agreement.csv',index=False)
    template=session/'ground_truth_events_template.csv'
    if not template.exists():
        template.write_text('reference_event_id,event_timestamp_s,travel_direction,plate_text,speed_kmh,violation_status\n')
    return df


def preview(session):
    """Encode a four-way preview from fully annotated pipeline videos."""
    import cv2
    import numpy as np
    import subprocess
    session=Path(session)
    names=['bytetrack','botsort','deepsort','tensorrt_fp16']
    caps=[cv2.VideoCapture(str(session/name/'run-01/annotated.mp4')) for name in names]
    try:
        if not all(c.isOpened() for c in caps):raise ValueError('A complete four-variant run-01 video set is required')
        fps=caps[0].get(cv2.CAP_PROP_FPS)
        counts=[int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps]
        if len(set(counts))!=1 or any(abs(c.get(cv2.CAP_PROP_FPS)-fps)>.001 for c in caps):
            raise ValueError('Preview inputs are not synchronized')
        import shutil
        ffmpeg=shutil.which('ffmpeg')
        if not ffmpeg:raise ValueError('Install ffmpeg before building preview')
        target=session/'preview_full_pipeline_4way.mp4'
        schedules=[]
        for name in names:
            directory=session/name/'run-01'
            with (directory/'events.csv').open(newline='') as handle:
                events={int(row['track_id']):row for row in csv.DictReader(handle)}
            schedule=defaultdict(list)
            with (directory/'detection_trace.jsonl').open() as handle:
                for line in handle:
                    record=json.loads(line)
                    if record.get('type')=='event_finalized':
                        schedule[record['finalized_at_frame']].append(events[record['track_id']])
            schedules.append(schedule)
        latest=[None]*len(names)
        process=subprocess.Popen([ffmpeg,'-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24',
            '-s','1920x1310','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast',
            '-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],stdin=subprocess.PIPE)
        labels=['ByteTrack / PT FP32','BoT-SORT GMC / PT FP32','DeepSORT MobileNet / PT FP32','ByteTrack / TRT detector FP16']
        try:
            for i in range(counts[0]):
                canvas=np.full((1310,1920,3),20,dtype=np.uint8)
                for j,cap in enumerate(caps):
                    ok,frame=cap.read()
                    if not ok:raise ValueError('Preview input decoding failed')
                    x=(j%2)*960;y=(j//2)*640
                    canvas[y+35:y+575,x:x+960]=cv2.resize(frame,(960,540))
                    cv2.putText(canvas,labels[j],(x+12,y+25),cv2.FONT_HERSHEY_SIMPLEX,.7,(240,240,240),2)
                    if schedules[j].get(i):latest[j]=schedules[j][i][-1]
                    event=latest[j]
                    if event:
                        lines=[f'Finalized event {event["event_id"]} | ID {event["track_id"]} | {event["speed_kmh"] or "N/A"} km/h | {event["violation_status"]}',
                               f'Hull: {event["plate_text"] or "unread"} | Class: {event["vehicle_class"]} | event t={event["event_timestamp_s"]}s']
                    else:lines=['No finalized business event yet','OCR and speed are computed by the original pipeline']
                    for k,line in enumerate(lines):
                        cv2.putText(canvas,line,(x+12,y+600+k*25),cv2.FONT_HERSHEY_SIMPLEX,.6,(220,235,245),1,cv2.LINE_AA)
                cv2.putText(canvas,f't={i/fps:.2f}s | Event details appear only after actual finalization | IDs are local',
                    (15,1300),cv2.FONT_HERSHEY_SIMPLEX,.6,(240,240,240),1)
                process.stdin.write(canvas.tobytes())
        finally:
            process.stdin.close()
        if process.wait()!=0:raise RuntimeError('ffmpeg preview failed')
        return target
    finally:
        for cap in caps:cap.release()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--variant',choices=['bytetrack','botsort','deepsort','pytorch_fp16','tensorrt_fp16'],required=True)
    parser.add_argument('--engine',type=Path)
    parser.add_argument('--max-seconds',type=float,default=45)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if not math.isfinite(args.max_seconds) or args.max_seconds<=0:parser.error('max-seconds must be positive')
    run_experiment(args.root,args.output,args.variant,args.engine,args.max_seconds,args.device)


if __name__=='__main__':main()
