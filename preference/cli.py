"""CLI shared by the Qt UI and unattended local jobs."""

import argparse
import json
import sys
from pathlib import Path

from .config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local SDXL preference collection and Diffusion-DPO")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "train"):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--stop-file", type=Path)
        if name == "train":
            command.add_argument("--resume", type=Path)
    inspect = sub.add_parser("status")
    inspect.add_argument("--dataset", required=True, type=Path)
    export = sub.add_parser("export")
    export.add_argument("--dataset", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    combined = sub.add_parser("combine", help="Export original adapters plus a DPO checkpoint as one inference LoRA")
    combined.add_argument("--config", required=True, type=Path)
    combined.add_argument("--preference", required=True, type=Path)
    combined.add_argument("--output", required=True, type=Path)
    combined.add_argument("--stop-file", type=Path)
    args = parser.parse_args(argv)
    if args.command in ("status", "export"):
        from .store import PreferenceStore
        store = PreferenceStore(args.dataset)
        if args.command == "export":
            store.export_jsonl(args.output)
        print(json.dumps(store.counts(), indent=2))
        return 0
    cfg = load_config(args.config)
    cancelled = lambda: bool(args.stop_file and args.stop_file.exists())
    if cancelled():
        print("Stop file already exists; no job started. Use a new stop-file path.")
        return 0
    if args.command == "combine":
        from .combined import export_combined
        try:
            export_combined(cfg, args.preference, args.output, cancelled)
        except InterruptedError as error:
            print(str(error), flush=True)
        return 0
    from .jobs import gpu_lock, Timeout
    lock = gpu_lock()
    try:
        with lock.acquire(timeout=0):
            from .sdxl import Cancelled
            try:
                if args.command == "generate":
                    from .generation import generate
                    generate(cfg, cancelled)
                else:
                    from .training import train
                    train(cfg, args.resume, cancelled)
            except Cancelled:
                print("Stopped at user request.", flush=True)
        return 0
    except Timeout:
        print("Another training or preference job is already running in this checkout.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        print(f"Configuration/data error: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
