"""Compare current and reference Co-Scientist run artifacts with DeepEval."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

EVAL_ROOT = Path(__file__).resolve().parents[1]
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from hypothesis_eval import ArtifactError, build_comparison_report, load_comparison_cases  # noqa: E402

ENV_FILE = EVAL_ROOT / ".env"
SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


def load_env_file(path: Path) -> None:
    """Load simple NAME=VALUE entries without overriding the process environment."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        if not separator or not name.strip():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(name.strip(), value)


def redact_secrets(message: str) -> str:
    redacted = message
    for name, value in os.environ.items():
        if value and len(value) >= 4 and any(marker in name.upper() for marker in SECRET_MARKERS):
            redacted = redacted.replace(value, "[REDACTED]")
    return redacted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-run", type=Path, action="append", required=True)
    parser.add_argument("--reference-run", type=Path, action="append", required=True)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--judge-model", default=None, help="defaults to LOCAL_MODEL_NAME")
    parser.add_argument("--judge-base-url", default=None, help="defaults to LOCAL_MODEL_BASE_URL")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--validate-only", action="store_true", help="parse inputs without contacting the judge")
    parser.add_argument("--quiet", action="store_true", help="hide DeepEval's per-case tables")
    return parser


def judge_config(args: argparse.Namespace) -> dict[str, str]:
    model = args.judge_model or os.environ.get("LOCAL_MODEL_NAME")
    base_url = args.judge_base_url or os.environ.get("LOCAL_MODEL_BASE_URL")
    api_key = os.environ.get("LOCAL_MODEL_API_KEY")
    if not model or not base_url or not api_key:
        raise ArtifactError(
            "judge requires LOCAL_MODEL_NAME, LOCAL_MODEL_BASE_URL, and LOCAL_MODEL_API_KEY in eval/.env"
        )
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ArtifactError("judge base URL must be an HTTP(S) URL without embedded credentials")
    return {"provider": "lm_studio", "model": model, "base_url": base_url, "api_key": api_key}


def default_report_path() -> Path:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    return EVAL_ROOT / "reports" / f"comparison-{timestamp}.json"


def main(argv: list[str] | None = None) -> int:
    load_env_file(ENV_FILE)
    args = build_parser().parse_args(argv)
    if not 0 <= args.threshold <= 1:
        print("Evaluation input error: threshold must be between 0 and 1", file=sys.stderr)
        return 2
    try:
        cases = load_comparison_cases(args.current_run, args.reference_run)
        if args.validate_only:
            print(
                json.dumps(
                    {
                        "status": "valid",
                        "research_goal": cases[0].goal,
                        "pairs": len(cases) // 2,
                        "runs": [
                            {
                                "system": case.system,
                                "run_id": case.run_id,
                                "source_step": case.source_step,
                                "hypothesis_id": case.hypothesis_id,
                            }
                            for case in cases
                        ],
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 0

        judge = judge_config(args)
        from deepeval.models import LocalModel

        from hypothesis_eval.deepeval_runner import evaluate_case

        model = LocalModel(
            model=judge["model"],
            base_url=judge["base_url"],
            api_key=judge["api_key"],
            temperature=0,
        )
        report_path = args.report or default_report_path()
        evaluations = [
            evaluate_case(
                case,
                model=model,
                threshold=args.threshold,
                results_folder=report_path.parent / "deepeval",
                print_results=not args.quiet,
            )
            for case in cases
        ]
        public_judge = {key: value for key, value in judge.items() if key != "api_key"}
        report = build_comparison_report(cases, evaluations, judge=public_judge, threshold=args.threshold)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except (ArtifactError, OSError, RuntimeError, ValueError) as exc:
        print(f"Evaluation error: {redact_secrets(str(exc))}", file=sys.stderr)
        return 2

    delta = report["delta_current_minus_reference"]["overall_mean"]
    print(f"Comparison complete: current - reference overall = {delta:+.4f}")
    print(f"Report: {report_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
