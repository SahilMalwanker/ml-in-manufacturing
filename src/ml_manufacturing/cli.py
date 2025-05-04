"""Command-line interface: ``ml-manufacturing <command> --help``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .datasets import DOMAINS, SCENARIOS, load_manifest, select
from .models import MODEL_NAMES

DEFAULT_MANIFEST = "data/manifest.csv"


def _add_mqtt_options(parser: argparse.ArgumentParser) -> None:
    from .realtime import DEFAULT_TOPIC, PASSWORD_ENV

    parser.add_argument("--host", default="localhost", help="MQTT broker host (default: localhost)")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port (default: 1883)")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help=f"MQTT topic (default: {DEFAULT_TOPIC})")
    parser.add_argument("--username", help=f"MQTT username; the password is read from ${PASSWORD_ENV}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml-manufacturing",
        description="Fault detection for the ST60 press station from real-plant and digital-twin data.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    convert = commands.add_parser("convert", help="label raw DAQ recordings and write CSV files")
    convert.add_argument("files", nargs="+", type=Path, help="raw DAQ .txt recordings")
    convert.add_argument("--fault", type=int, required=True, help="fault code injected in these recordings")
    convert.add_argument("--delay-ms", type=float, help="scripted delay between acquisition start and fault")
    convert.add_argument("--method", choices=("auto", "rule", "delay", "whole"), default="auto")
    convert.add_argument("--out-dir", type=Path, default=Path("data/labelled"))

    train = commands.add_parser("train", help="train a detector on the training split of a domain")
    train.add_argument("--manifest", type=Path, default=Path(DEFAULT_MANIFEST))
    train.add_argument("--domain", choices=(*DOMAINS, "combined"), default="combined")
    train.add_argument("--model", choices=MODEL_NAMES, default="svc")
    train.add_argument("--window", type=int, default=3, help="samples per window (0 = no windowing)")
    train.add_argument("--out", type=Path, default=Path("models/detector.joblib"))

    evaluate = commands.add_parser("evaluate", help="evaluate a trained detector on a test split")
    evaluate.add_argument("--model", type=Path, required=True, help="detector bundle from 'train'")
    evaluate.add_argument("--manifest", type=Path, default=Path(DEFAULT_MANIFEST))
    evaluate.add_argument("--domain", choices=(*DOMAINS, "combined"), default="combined")
    evaluate.add_argument("--split", choices=("train", "test"), default="test")
    evaluate.add_argument("--confusion-matrix", type=Path, help="write a confusion-matrix PNG")

    bench = commands.add_parser("benchmark", help="all scenario x model x window combinations")
    bench.add_argument("--manifest", type=Path, default=Path(DEFAULT_MANIFEST))
    bench.add_argument("--scenarios", nargs="+", choices=tuple(SCENARIOS), default=list(SCENARIOS))
    bench.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    bench.add_argument("--windows", nargs="+", type=int, default=[0, 3, 5, 10])
    bench.add_argument("--seed", type=int, default=42, help="random state of the classifiers (default: 42)")
    bench.add_argument("--no-clip", action="store_true", help="do not clip scaled inputs to the training range")
    bench.add_argument("--out", type=Path, default=Path("results/benchmark.csv"))

    serve = commands.add_parser("serve", help="real-time fault detection on the MQTT stream")
    serve.add_argument("--model", type=Path, required=True, help="detector bundle from 'train'")
    _add_mqtt_options(serve)
    serve.add_argument("--publish-topic", help="also publish predictions as JSON to this topic")
    serve.add_argument("--no-dedupe", action="store_true", help="keep payloads whose signals did not change")
    serve.add_argument("--changes-only", action="store_true", help="print only when the predicted class changes")

    replay = commands.add_parser("replay", help="publish recordings to MQTT like the PLC does")
    replay.add_argument("files", nargs="+", type=Path, help="recordings (.txt raw DAQ or labelled .csv)")
    _add_mqtt_options(replay)
    replay.add_argument("--speed", type=float, default=1.0, help="playback speed factor (0 = no delay)")
    replay.add_argument("--interval-ms", type=float, default=100.0, help="pacing for files without timestamps")
    return parser


def _convert(args: argparse.Namespace) -> int:
    from .daq import label_recording, read_recording, to_labelled_csv

    for path in args.files:
        frame = read_recording(path)
        labels = label_recording(frame, args.fault, args.delay_ms, args.method)
        target = to_labelled_csv(frame, labels, args.out_dir / f"{path.stem}.csv")
        faulty = int((labels != 0).sum())
        print(f"{path.name}: {len(labels)} samples, {faulty} labelled {args.fault} -> {target}")
    return 0


def _train(args: argparse.Namespace) -> int:
    from .detector import FaultDetector

    recordings = select(load_manifest(args.manifest), args.domain, "train")
    if not recordings:
        print(f"no training recordings for domain '{args.domain}' in {args.manifest}", file=sys.stderr)
        return 1
    detector = FaultDetector(args.model, args.window).fit([rec.load() for rec in recordings])
    detector.save(args.out)
    print(
        f"trained {args.model.upper()} (window {args.window}) on {len(recordings)} {args.domain} recordings "
        f"in {detector.fit_seconds:.3f} s; classes {detector.classes} -> {args.out}"
    )
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from .detector import FaultDetector

    detector = FaultDetector.load(args.model)
    recordings = select(load_manifest(args.manifest), args.domain, args.split, faults=set(detector.classes))
    if not recordings:
        print(f"no {args.split} recordings for domain '{args.domain}' in {args.manifest}", file=sys.stderr)
        return 1
    result = detector.evaluate([rec.load() for rec in recordings])
    print(f"{len(recordings)} recordings, {len(result.y_true)} windows")
    print(
        f"accuracy {result.accuracy:.4f} | precision {result.precision:.4f} | "
        f"recall {result.recall:.4f} | F1 {result.f1:.4f} (macro)"
    )
    print(result.report())
    if args.confusion_matrix:
        from .plotting import plot_confusion

        print(f"confusion matrix -> {plot_confusion(result, args.confusion_matrix)}")
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    from .benchmark import run_benchmark

    table = run_benchmark(
        load_manifest(args.manifest), args.scenarios, args.models, args.windows, args.seed, clip=not args.no_clip
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"{len(table)} results -> {args.out}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    from .realtime import serve

    serve(
        str(args.model),
        host=args.host,
        port=args.port,
        topic=args.topic,
        username=args.username,
        publish_topic=args.publish_topic,
        dedupe=not args.no_dedupe,
        changes_only=args.changes_only,
    )
    return 0


def _replay(args: argparse.Namespace) -> int:
    from .replay import replay

    replay(args.files, args.host, args.port, args.topic, args.speed, args.interval_ms, args.username)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "convert": _convert,
        "train": _train,
        "evaluate": _evaluate,
        "benchmark": _benchmark,
        "serve": _serve,
        "replay": _replay,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
