import argparse
import os
import sys

from .app import CameraError, run

DEFAULT_MODEL = os.path.join("models", "hand_landmarker.task")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="airscribe", description="Write and draw in the air with your index finger.")
    p.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    p.add_argument("--width", type=int, default=1280, help="requested frame width (default 1280)")
    p.add_argument("--height", type=int, default=720, help="requested frame height (default 720)")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"path to hand_landmarker.task (default {DEFAULT_MODEL})")
    p.add_argument("--export-dir", default="exports", help="folder for saved PNGs (default exports)")
    p.add_argument("--no-mirror", action="store_true", help="start with the true (unmirrored) camera view; toggle with M")
    p.add_argument("--log", metavar="CSV", help="write a per-frame gesture log for tuning")
    args = p.parse_args(argv)
    try:
        run(args.camera, args.width, args.height, args.model, args.export_dir, mirror=not args.no_mirror, log_path=args.log)
    except (CameraError, FileNotFoundError) as e:
        print(f"AirScribe error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
