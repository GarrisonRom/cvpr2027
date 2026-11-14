"""Replay recorded actions without a policy and audit camera repeatability."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"src"))
from safety_continuity.evaluation.render_diagnostic import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--render-repeats", type=int, default=16)
    parser.add_argument("--render-ablations", action="store_true",
                        help="At held step 5, compare shadow/reflection/MSAA settings with native controls")
    parser.add_argument("--render-profile", choices=("native", "egl_no_msaa_v1"), default="native")
    args = parser.parse_args()
    run(args.source_run.resolve(), args.output_root.resolve(), steps=args.steps,
        repeats=args.repeats, render_repeats=args.render_repeats, render_ablations=args.render_ablations,
        render_profile=args.render_profile)
