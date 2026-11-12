# SPDX-License-Identifier: Apache-2.0
"""rashomon-llm command line.

Argument surface only; the protocol is described in docs/protocol.md.
"""
from __future__ import annotations

import argparse
import sys

from rashomon_llm import __version__


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rashomon",
        description=(
            "Measure how often an LLM endpoint returns different outcomes for "
            "identical request parameters under concurrent load. It measures; "
            "it does not fix, and it does not rank models."
        ),
    )
    p.add_argument("--version", action="version", version=f"rashomon-llm {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a measurement against an endpoint")
    run.add_argument("--base-url", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--api-key-env", default=None, help="environment variable holding the API key")
    run.add_argument("--set", dest="set_name", required=True, help="agent40 | math40 | path to a set")
    run.add_argument("--n", type=int, default=8, help="sweeps per set (copies per prompt)")
    run.add_argument("--k", type=int, default=8, help="max measured requests in flight")
    run.add_argument("--seed", type=int, default=1234)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--top-p", type=float, default=None)
    run.add_argument("--top-k", type=int, default=None)
    run.add_argument("--min-p", type=float, default=None)
    run.add_argument("--max-tokens", type=int, default=2048)
    run.add_argument("--mode", choices=("sweep", "burst"), default="sweep",
                     help="sweep is the estimand; burst is a labelled diagnostic")
    run.add_argument("--sampling", choices=("fixed", "production"), default="fixed",
                     help="production sampling is reported separately, never mixed")
    run.add_argument("--grader", default=None, help="local outcome-function script (stdin JSON -> string)")
    run.add_argument("--outcome", choices=("agent", "math", "none"), default=None,
                     help="outcome function; default: inferred from the set name (agent*/math*/long*)")
    run.add_argument("--control", action="store_true", help="also run the lower-concurrency (K=1) control arm")
    run.add_argument("--control-n", type=int, default=None,
                     help="sweeps for the K=1 control arm (default: same as --n)")
    run.add_argument("--compare-endpoint", default=None, help="second base URL for a labelled comparison")
    run.add_argument("--engine", default=None, help="engine name, user-reported (unverified)")
    run.add_argument("--hardware", default=None, help="hardware description, user-reported (unverified)")
    run.add_argument("--no-probe", action="store_true", help="skip the read-only server metadata probe")
    run.add_argument("--record-base-url", action="store_true",
                     help="store the full base URL in run.json (default: scheme + truncated host sha256)")
    run.add_argument("--quick", action="store_true", help="512 tokens, no control; labelled not comparable")
    run.add_argument("--out", required=True, help="path to run.json")

    rep = sub.add_parser("report", help="render JSON + Markdown + card from a run file")
    rep.add_argument("run_json")
    rep.add_argument("--grader", default=None,
                     help="outcome-function script; required for a run that used one (a grader named "
                          "in the run file is never executed; its recorded sha256 must match)")
    rep.add_argument("--compare", default=None,
                     help="second run file; adds a labelled side-by-side (score + tok/s)")
    rep.add_argument("--model-label", default=None,
                     help="display name for the model on the card and in the Markdown "
                          "(display only; not part of report_sha256)")

    ver = sub.add_parser("verify", help="recompute the report from a run file and compare its sha256")
    ver.add_argument("run_json")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        from .cli_run import run_command
        return run_command(args)
    if args.command == "report":
        from .cli_report import report_command
        return report_command(args)
    if args.command == "verify":
        from .cli_report import verify_command
        return verify_command(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
