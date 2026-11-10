"""Video-time orchestration; inference can be replaced by a test adapter."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
import time

from hauling.config import read_config
from hauling.vision.detection import YoloAdapter, local_weights
from hauling.io.evidence import HullEvidence
from hauling.engine.finish_window import FinishWindows
from hauling.utils.geometry import centroid, contains, scale_polygon
from hauling.vision.hull_history import HullReadHistory
from hauling.utils.offline import offline_runtime
from hauling.io.output import EventWriter, event_record
from hauling.io.report import TrackReport
from hauling.io.render import AnnotatedVideo, FrameBuffer, annotate
from hauling.vision.recognition import CharacterModels, HullRecognizer
from hauling.engine.state import TrackEngine, VehicleEvent
from hauling.io.trace import DetectionTrace
from hauling.vision.vehicle_type import VehicleTypeResolver

LOGGER = logging.getLogger("hauling")


@dataclass(frozen=True)
class RunOptions:
    video: Path
    vehicle_model: Path
    plate_model: Path | None
    config: Path
    output_dir: Path
    save_video: bool = False
    device: str = "auto"
    max_frames: int | None = None
    max_seconds: float | None = None
    tracker_config: Path = Path(__file__).resolve().parents[1] / "bytetrack.yaml"
    tracker_buffer_seconds: float = 1.0
    plate_confidence: float = 0.25
    threads: int = 4
    hull_min_aspect_ratio: float = 2.0
    hull_reference: Path | None = None
    save_hull_crops: bool = False
    log_detections: bool = False
    character_models: CharacterModels | None = None
    character_confidence: float = 0.5
    vehicle_classifier: Path | None = None
    light_max_width_ratio: float = .30
    ocr_sample_seconds: float = .2
    ocr_lookahead_seconds: float = .3
    report: bool = False


def run_pipeline(options: RunOptions, adapter_factory=YoloAdapter) -> dict:
    cfg = read_config(options.config)
    if not math.isfinite(options.ocr_lookahead_seconds) or options.ocr_lookahead_seconds < 0:
        raise ValueError("OCR look-ahead must be a nonnegative finite number")
    if not options.video.is_file():
        raise ValueError("Video must be an existing local file")
    local_weights(options.vehicle_model)
    if options.plate_model:
        local_weights(options.plate_model)
    if options.vehicle_classifier is not None:
        local_weights(options.vehicle_classifier)
    if options.character_models is not None:
        if options.plate_model is None:
            raise ValueError("Character recognition requires hull detection")
        options.character_models.validate()
    if options.hull_reference is not None and options.plate_model is None:
        raise ValueError("A hull reference requires hull detection to be enabled")
    if options.hull_reference is not None and not options.hull_reference.is_file():
        raise ValueError("Hull reference must be an existing local image")
    options.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("events.csv", "annotated.mp4", "run_summary.json"):
        if (options.output_dir / name).exists():
            raise ValueError("Output artifacts already exist; choose a new --output-dir")
    if options.save_hull_crops and (options.output_dir / "hull_crops").exists():
        raise ValueError("Hull crop artifacts already exist; choose a new --output-dir")
    if options.log_detections and (options.output_dir / "detection_trace.jsonl").exists():
        raise ValueError("Detection trace already exists; choose a new --output-dir")
    if options.report and (options.output_dir / "report").exists():
        raise ValueError("Report artifacts already exist; choose a new --output-dir")

    started = time.perf_counter()
    summary = {
        "source_video": options.video.name,
        "status": "running",
        "fps": None,
        "resolution": None,
        "source_frame_count": None,
        "total_frames_read": 0,
        "total_detections": 0,
        "unique_tracks": 0,
        "eligible_tracks": 0,
        "events_csv": 0,
        "speed_valid": 0,
        "overspeed": 0,
        "plate_detected": 0,
        "plate_detection_enabled": options.plate_model is not None,
        "identification_target": "hull_number_above_front_windshield",
        "plate_inference_errors": 0,
        "tracks_without_final_event": 0,
        "video_saved": False,
        "reached_eof": False,
        "partial_run": False,
        "processing_duration_seconds": 0,
        "distance_meter": cfg.distance_meter,
        "speed_limit_kmh": cfg.speed_limit_kmh,
        "reference_travel_direction": cfg.travel_direction,
        "vehicle_model": options.vehicle_model.name,
        "plate_model": options.plate_model.name if options.plate_model else None,
        "warnings": [],
        "errors": [],
        "hull_reference": options.hull_reference.name if options.hull_reference else None,
        "hull_reference_method": "red_on_white_panel_with_glass_context" if options.hull_reference else None,
        "hull_reference_score": "appearance_score_not_probability" if options.hull_reference else None,
        "hull_crop_events_saved": 0,
        "detection_trace_enabled": options.log_detections,
        "event_trigger": "first_final_area_contact",
        "hull_inference_scope": "sampled_approach_and_finish_output_at_finish_only",
        "character_recognition_enabled": options.character_models is not None,
        "character_recognition_errors": 0,
        "hull_text_recognized": 0,
        "hull_text_complete": 0,
        "vehicle_type_refinement_enabled": options.vehicle_classifier is not None,
        "vehicle_type_refinement_errors": 0,
        "main_area_policy": "admit_inside_then_preserve_track_continuity_to_final",
        "ocr_lookahead_seconds": options.ocr_lookahead_seconds,
        "ocr_windows_truncated": 0,
        "report_enabled": options.report,
        "report_saved": False,
        "track_photos_saved": 0,
        "report_errors": 0,
    }
    capture = csv_writer = video_writer = engine = adapter = recognizer = None
    trace = None
    vehicle_resolver = None
    histories = {}
    last_hull_frame = {}
    finalized_plates = {}
    final_events = {}
    video_buffer = None
    failure = None
    track_ids = set()
    evidence = HullEvidence(options.output_dir / "hull_crops") if options.save_hull_crops else None
    report = TrackReport(options.output_dir / "report") if options.report else None

    def warn(message: str) -> None:
        if message not in summary["warnings"]:
            summary["warnings"].append(message)
            LOGGER.warning(message)

    try:
        with offline_runtime(options.output_dir):
            import cv2

            capture = cv2.VideoCapture(str(options.video.resolve()))
            if not capture.isOpened():
                raise ValueError("Cannot open video; verify the local file and codec")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError("Video has no valid FPS; video-time speed cannot be calculated")
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError("Cannot decode the first video frame")
            height, width = frame.shape[:2]
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            summary.update(fps=fps, resolution=[width, height], source_frame_count=frame_count)
            areas = {name: scale_polygon(area, cfg.resolution, (width, height)) for name, area in cfg.areas.items()}
            engine = TrackEngine(cfg, areas, max(1, int(options.tracker_buffer_seconds * fps)))
            hull_interval = max(1, round(options.ocr_sample_seconds * fps))
            lookahead_frames = math.floor(options.ocr_lookahead_seconds * fps + 1e-9) if options.plate_model else 0
            windows = FinishWindows(lookahead_frames)
            summary["ocr_lookahead_frames"] = lookahead_frames
            summary["hull_inference_scope"] = "sampled_approach_and_bounded_finish_window_event_at_final"
            if options.log_detections:
                trace = DetectionTrace(options.output_dir / "detection_trace.jsonl", areas, fps,
                                       engine.max_crossing_gap_frames)
            adapter = adapter_factory(
                options.vehicle_model, options.plate_model, cfg, fps, options.device,
                options.tracker_config, options.tracker_buffer_seconds,
                options.plate_confidence, options.threads,
                options.hull_min_aspect_ratio,
                options.hull_reference,
            )
            summary["device"] = adapter.device
            summary["model_classes"] = json.loads(json.dumps(adapter.model_classes))
            if options.vehicle_classifier is not None:
                vehicle_resolver = VehicleTypeResolver(options.vehicle_classifier, adapter.device,
                                                       areas["main"], round(fps), options.light_max_width_ratio)
                summary["vehicle_classifier"] = options.vehicle_classifier.name
                summary["light_max_width_ratio"] = options.light_max_width_ratio
            if options.character_models is not None:
                recognizer = HullRecognizer(options.character_models, adapter.device, options.character_confidence)
                summary["character_model_classes"] = recognizer.model_classes
                summary["character_models"] = {name: path.name for name, path in vars(options.character_models).items()}
                summary["character_confidence"] = options.character_confidence
            summary["inference_settings"] = {
                "confidence": cfg.confidence, "iou": cfg.iou,
                "image_size": cfg.image_size, "max_detections": cfg.max_detections,
                "agnostic_nms": cfg.agnostic_nms,
                "tracker_buffer_seconds": options.tracker_buffer_seconds,
                "plate_confidence": options.plate_confidence,
                "hull_min_aspect_ratio": options.hull_min_aspect_ratio,
                "crossing_detection": "observed_centroid_segments_within_tracker_buffer",
            }
            if options.plate_model is None:
                warn("Hull-number detection explicitly disabled; hull-model acceptance remains unvalidated")
            csv_writer = EventWriter(options.output_dir / "events.csv")
            if options.save_video:
                try:
                    video_writer = AnnotatedVideo(options.output_dir / "annotated.mp4", fps, (width, height))
                    video_buffer = FrameBuffer(lookahead_frames)
                except Exception:
                    warn("Cannot initialize annotated MP4 writer; CSV processing continues")

            def finalize_events(frame_index: int, force: bool = False) -> None:
                for pending in windows.ready(frame_index, force):
                    event = pending.event
                    state = event.track
                    if force and frame_index < pending.deadline_frame:
                        state.ocr_window_truncated = True
                        summary["ocr_windows_truncated"] += 1
                    if state.track_id in histories:
                        state.plate = histories[state.track_id].resolve()
                        if evidence and state.plate is not None:
                            evidence.finalize(state.track_id, state.plate)
                    finalized_plates[state.track_id] = state.plate
                for track_id, event in list(final_events.items()):
                    if track_id not in finalized_plates:
                        continue
                    state = event.track
                    record = event_record(event, cfg, options.video.name, str(summary["events_csv"] + 1))
                    csv_writer.write(record)
                    state.event_written = True
                    state.event_pending = False
                    del final_events[track_id]
                    if report:
                        try:
                            report.save(record)
                        except (OSError, ValueError):
                            summary["report_errors"] += 1
                            warn("Cannot save report vehicle photo; CSV processing continues")
                    if evidence:
                        try:
                            summary["hull_crop_events_saved"] += int(evidence.save(record["event_id"], state.track_id))
                        except OSError:
                            warn("Cannot save hull crop evidence; CSV processing continues")
                    summary["events_csv"] += 1
                    summary["speed_valid"] += int(event.measurement.valid)
                    summary["overspeed"] += int(event.measurement.is_overspeed)
                    summary["plate_detected"] += int(state.plate is not None)
                    summary["hull_text_recognized"] += int(bool(state.plate and state.plate.text))
                    summary["hull_text_complete"] += int(bool(state.plate and state.plate.text_status == "complete"))
                    LOGGER.info("event_written track=%s t=%.6f status=%s", state.track_id,
                                event.timestamp_s, event.measurement.status)
                    if trace:
                        trace.emit({"type": "event_finalized", "track_id": state.track_id,
                                    "event_frame_index": event.frame_index, "finalized_at_frame": frame_index})

            def write_buffered(force: bool = False) -> None:
                nonlocal video_writer, video_buffer
                if not video_writer or video_buffer is None:
                    return
                try:
                    for buffered in video_buffer.ready(force):
                        for track_id, state in buffered.engine.tracks.items():
                            if state.finish_s == buffered.timestamp and track_id in finalized_plates:
                                state.plate = finalized_plates[track_id]
                        video_writer.write(annotate(buffered.frame, buffered.detections, buffered.engine,
                                                    areas, cfg, buffered.timestamp, options.plate_model is not None,
                                                    buffered.plates))
                except Exception:
                    try:
                        video_writer.close()
                    except Exception:
                        pass
                    video_writer = video_buffer = None
                    warn("Annotated video rendering/writing failed; CSV processing continues")
            limit = options.max_frames
            if options.max_seconds is not None:
                seconds_limit = max(1, math.ceil(options.max_seconds * fps))
                limit = min(limit, seconds_limit) if limit is not None else seconds_limit
            while ok:
                index = summary["total_frames_read"]
                timestamp_s = index / fps
                summary["total_frames_read"] += 1
                if frame.shape[:2] != (height, width):
                    raise ValueError("Video resolution changed during decoding")
                try:
                    detections, count = adapter.detect(frame)
                except Exception as error:
                    raise RuntimeError(
                        f"Vehicle inference failed at frame {index} ({type(error).__name__}); "
                        "check model compatibility and device settings"
                    ) from None
                summary["total_detections"] += count
                if vehicle_resolver is not None:
                    refined = []
                    for detection in detections:
                        previous = engine.tracks.get(detection.track_id)
                        try:
                            refined.append(vehicle_resolver.resolve(frame, detection, index,
                                           frozen=bool(previous and previous.finish_s is not None)))
                        except Exception as error:
                            refined.append(detection)
                            summary["vehicle_type_refinement_errors"] += 1
                            warn(f"Vehicle type refinement failed ({type(error).__name__}); detector class retained")
                    detections = refined
                current_plates = {}
                if trace:
                    trace.begin_frame(index, timestamp_s, detections, count)
                for detection in detections:
                    trace_record = trace.before(index, timestamp_s, detection, engine) if trace else None
                    if detection.track_id is None:
                        if trace:
                            trace.after(trace_record, engine, "unassigned")
                        continue
                    track_ids.add(detection.track_id)
                    point = centroid(detection.bbox_xyxy)
                    if detection.track_id not in engine.tracks and not contains(areas["main"], point):
                        if trace:
                            trace.after(trace_record, engine, "outside_main")
                        continue
                    previous = engine.tracks.get(detection.track_id)
                    if previous is not None and previous.event_written and not windows.contains(detection.track_id):
                        if trace:
                            trace.after(trace_record, engine, "event_already_written")
                        continue
                    before = (previous.start_s, previous.finish_s) if previous else (None, None)
                    event = engine.observe(detection.track_id, point, detection.confidence,
                                           timestamp_s, index, vehicle_class_id=detection.class_id,
                                           vehicle_class=detection.class_name,
                                           vehicle_class_source=detection.class_source)
                    state = engine.tracks[detection.track_id]
                    finish_contact = before[1] is None and state.finish_s is not None
                    if finish_contact:
                        windows.add(VehicleEvent(state, timestamp_s, index, state.measurement(cfg)))
                    if event:
                        final_events[state.track_id] = event
                        if report:
                            try:
                                report.remember(frame, detection, event)
                            except (OSError, ValueError):
                                summary["report_errors"] += 1
                                warn("Cannot capture report vehicle photo; CSV processing continues")
                        if state.finish_s is None:
                            # No speed measurement: finalize a bounded empty/approach OCR result.
                            windows.add(event)
                    plate = None
                    due = index - last_hull_frame.get(detection.track_id, -hull_interval) >= hull_interval
                    if options.plate_model is not None and (finish_contact or windows.contains(detection.track_id) or (state.finish_s is None and not state.event_pending and state.start_s is not None and due)):
                        last_hull_frame[detection.track_id] = index
                        try:
                            plate = adapter.detect_plate(frame, detection, timestamp_s)
                            if plate is not None:
                                if not plate.valid or plate.timestamp_s != timestamp_s:
                                    raise ValueError("Hull candidate must belong to its observed source frame")
                                if recognizer is not None:
                                    try:
                                        plate = recognizer.recognize(frame, plate)
                                    except Exception as error:
                                        summary["character_recognition_errors"] += 1
                                        warn(f"Hull character recognition failed ({type(error).__name__}); hull region retained")
                                history = histories.setdefault(detection.track_id, HullReadHistory())
                                history.add(plate)
                                if evidence and history.best is plate:
                                    evidence.remember(frame, detection, plate)
                                if finish_contact:
                                    current_plates[detection.track_id] = plate
                        except Exception as error:
                            summary["plate_inference_errors"] += 1
                            warn(f"Hull-region inference failed ({type(error).__name__}); vehicle events continue")
                    if before[0] is None and state.start_s is not None:
                        LOGGER.info("entered_start track=%s t=%.6f", state.track_id, timestamp_s)
                    if before[1] is None and state.finish_s is not None:
                        LOGGER.info("entered_finish track=%s t=%.6f", state.track_id, timestamp_s)
                    if trace:
                        if event:
                            gate = "final_pending_persistence"
                        else:
                            gate = "ocr_lookahead" if windows.contains(detection.track_id) and state.event_written else "observed"
                        trace.after(trace_record, engine, gate, False)
                finalize_events(index)
                if trace:
                    trace.flush()
                if video_writer:
                    video_buffer.push(frame, detections, engine, timestamp_s, current_plates)
                    write_buffered()
                if summary["total_frames_read"] % 250 == 0:
                    LOGGER.info("processed_frames=%s video_time=%.2fs", summary["total_frames_read"], timestamp_s)
                if limit is not None and summary["total_frames_read"] >= limit:
                    summary["partial_run"] = frame_count <= 0 or summary["total_frames_read"] < frame_count
                    summary["reached_eof"] = not summary["partial_run"]
                    break
                ok, frame = capture.read()
                if not ok:
                    summary["reached_eof"] = True
                    if frame_count > summary["total_frames_read"]:
                        summary["reached_eof"] = False
                        warn("Video decoding stopped before the declared frame count; input may be truncated")
            finalize_events(summary["total_frames_read"] - 1, force=True)
            write_buffered(force=True)
            summary["status"] = "completed"
    except BaseException as error:
        failure = error
        summary["status"] = "failed"
        # Exceptions are sanitized at config/model boundaries. Other libraries
        # only see frames and local file paths, never production configuration.
        summary["errors"].append(str(error) if isinstance(error, (ValueError, OSError, RuntimeError))
                                 else type(error).__name__)
    finally:
        if capture is not None:
            capture.release()
        if trace:
            summary["detection_trace_frames"] = trace.frames
            summary["detection_trace_rows"] = trace.detections
            try:
                trace.close()
            except OSError:
                summary["errors"].append("Detection trace could not close successfully")
                summary["status"] = "failed"
                if failure is None:
                    failure = OSError("Detection trace could not close successfully")
        if csv_writer:
            try:
                csv_writer.close()
            except Exception:
                summary["status"] = "failed"
                summary["errors"].append("CSV writer could not close successfully")
                if failure is None:
                    failure = OSError("CSV writer could not close successfully")
        if video_writer:
            try:
                video_writer.close()
                summary["video_saved"] = video_writer.verify()
                if not summary["video_saved"]:
                    warn("Annotated MP4 verification failed; CSV results remain available")
            except Exception:
                warn("Cannot finalize annotated MP4; CSV results remain available")
        summary["unique_tracks"] = len(track_ids)
        if vehicle_resolver is not None:
            summary["vehicle_type_diagnostics"] = vehicle_resolver.diagnostics
        if adapter is not None and hasattr(adapter, "hull_diagnostics"):
            summary["hull_detection_diagnostics"] = dict(adapter.hull_diagnostics)
        if adapter is not None and hasattr(adapter, "reference_diagnostics"):
            summary["hull_reference_diagnostics"] = dict(adapter.reference_diagnostics)
        if evidence:
            try:
                evidence.close()
            except OSError:
                warn("Cannot save hull crop manifest; CSV results remain available")
        if engine:
            summary["eligible_tracks"] = len(engine.tracks)
            summary["tracks_without_final_event"] = sum(not state.event_written for state in engine.tracks.values())
        summary["processing_duration_seconds"] = round(time.perf_counter() - started, 3)
        if report:
            summary["track_photos_saved"] = report.photos_saved
            try:
                report.close(summary)
                summary["report_saved"] = True
            except (OSError, ValueError):
                summary["report_errors"] += 1
                warn("Cannot finalize vehicle report; CSV results remain available")
        with (options.output_dir / "run_summary.json").open("x", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
    if failure:
        raise failure
    return summary
