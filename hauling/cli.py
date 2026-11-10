import argparse
import logging
import math
from pathlib import Path
import sys

from hauling.engine.pipeline import RunOptions, run_pipeline
from hauling.vision.recognition import CharacterModels


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def confidence(value: str) -> float:
    number = positive_float(value)
    if number > 1:
        raise argparse.ArgumentTypeError("confidence must be in (0, 1]")
    return number


def nonnegative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative finite number")
    return number


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline hauling video inference; no network services")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--vehicle-model", type=Path, required=True)
    plate = parser.add_mutually_exclusive_group(required=True)
    plate.add_argument("--hull-model", "--plate-model", dest="plate_model", type=Path,
                       help="Local hull-number region detector (No_Lambung)")
    plate.add_argument("--no-hull", "--no-plate", dest="no_plate", action="store_true",
                       help="Explicitly disable hull-number detection for partial validation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder without previous output artifacts")
    parser.add_argument("--render", "--save-video", dest="save_video", action="store_true",
                        help="Render annotated.mp4")
    parser.add_argument("--report", action="store_true",
                        help="Write a per-event vehicle report and photos at final-area contact")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, or intel:gpu for OpenVINO")
    parser.add_argument("--max-frames", type=positive_int, help="Process only a prefix for debugging")
    parser.add_argument("--max-seconds", type=positive_float, help="Process a prefix using video time")
    parser.add_argument("--tracker-config", type=Path, default=Path(__file__).with_name("bytetrack.yaml"))
    parser.add_argument("--tracker-buffer-seconds", type=positive_float, default=1.0)
    parser.add_argument("--hull-confidence", "--plate-confidence", dest="plate_confidence", type=confidence, default=0.25)
    parser.add_argument("--hull-min-aspect-ratio", type=positive_float, default=2.0,
                        help="Minimum width/height of the horizontal hull-number strip")
    parser.add_argument("--threads", type=positive_int, default=4, help="PyTorch CPU inference threads")
    parser.add_argument("--hull-reference", type=Path,
                        help="Optional red-on-white panel example for appearance/context fallback; no OCR or training")
    parser.add_argument("--save-hull-crops", action="store_true",
                        help="Save each event's selected hull crop, vehicle context, and JSON manifest")
    parser.add_argument("--log-detections", action="store_true",
                        help="Print every detection/ID and save detailed detection_trace.jsonl, including ROI skips")
    parser.add_argument("--no-ocr", action="store_true", help="Disable hull letter/digit recognition")
    models = Path("model-requirement")
    parser.add_argument("--vehicle-classifier", type=Path,
                        default=models / "Classification_truck_mitra_yolov8m_Imgz224_V3.pt")
    parser.add_argument("--no-vehicle-refinement", action="store_true",
                        help="Use detector classes without secondary camera-specific refinement")
    parser.add_argument("--light-max-width-ratio", type=confidence, default=.30,
                        help="Maximum vehicle/main-ROI width ratio for a non-coal Light Vehicle")
    parser.add_argument("--digit-detector", type=Path, default=models / "character_angka_detection_hauling_yolov8m_V2.pt")
    parser.add_argument("--letter-detector", type=Path, default=models / "character_huruf_detection_hauling_yolov8m_V2.pt")
    parser.add_argument("--digit-classifier", type=Path, default=models / "classification_hauling_angka_yolo8m_imgz96_v9.pt")
    parser.add_argument("--letter-classifier", type=Path, default=models / "classification_hauling_huruf_yolo8m_imgz96_v9.pt")
    parser.add_argument("--character-confidence", type=confidence, default=0.5,
                        help="Minimum letter/digit classification confidence")
    parser.add_argument("--ocr-sample-seconds", type=positive_float, default=.2,
                        help="Approach-frame OCR sampling interval; events are recorded at final-area contact")
    parser.add_argument("--ocr-lookahead-seconds", type=nonnegative_float, default=.3,
                        help="Offline OCR window after speed finish; events remain at final-area contact (0 disables)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    options = RunOptions(
        video=args.video, vehicle_model=args.vehicle_model, plate_model=args.plate_model,
        config=args.config, output_dir=args.output_dir, save_video=args.save_video, report=args.report,
        device=args.device, max_frames=args.max_frames, max_seconds=args.max_seconds,
        tracker_config=args.tracker_config, tracker_buffer_seconds=args.tracker_buffer_seconds,
        plate_confidence=args.plate_confidence, threads=args.threads,
        hull_min_aspect_ratio=args.hull_min_aspect_ratio,
        hull_reference=args.hull_reference, save_hull_crops=args.save_hull_crops,
        log_detections=args.log_detections,
        character_models=None if args.no_ocr or args.no_plate else CharacterModels(
            args.digit_detector, args.letter_detector, args.digit_classifier, args.letter_classifier),
        character_confidence=args.character_confidence,
        vehicle_classifier=None if args.no_vehicle_refinement else args.vehicle_classifier,
        light_max_width_ratio=args.light_max_width_ratio,
        ocr_sample_seconds=args.ocr_sample_seconds,
        ocr_lookahead_seconds=args.ocr_lookahead_seconds,
    )
    try:
        summary = run_pipeline(options)
    except KeyboardInterrupt:
        print("Interrupted; check run_summary.json for partial results", file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except ImportError:
        print("Error: missing inference dependency; install the project with pip install -e '.[test]'", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Error: {type(error).__name__}; check dependencies and run_summary.json", file=sys.stderr)
        return 1
    print(f"Completed: {summary['total_frames_read']} frames, {summary['events_csv']} events, "
          f"{summary['speed_valid']} valid speeds. Outputs: {options.output_dir}")
    return 0
