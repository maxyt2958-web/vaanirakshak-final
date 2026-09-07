"""Entry point for the package.

Three commands:

* ``python run.py demo``        — offline, no server, no GPU, no audio.
* ``python run.py serve``       — start the FastAPI server (WS + REST).
* ``python run.py eval``        — reproduce the ASVspoof 5 metric
                                  numbers on a synthetic stream.
* ``python run.py bench``       — measure end-to-end latency p50/p95.
"""

from __future__ import annotations

import argparse
import sys


def cmd_demo(_args: argparse.Namespace) -> int:
    from .demo import run_demo
    return run_demo()


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn
    from ..config import settings

    host = args.host or settings.server.host
    port = args.port or settings.server.port
    uvicorn.run("vanirakshak.server.app:app", host=host, port=port, reload=args.reload, log_level="info")
    return 0


def cmd_eval(_args: argparse.Namespace) -> int:
    from .eval_synth import run_eval
    return run_eval()


def cmd_bench(_args: argparse.Namespace) -> int:
    from .bench import run_bench
    return run_bench()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vanirakshak", description="VaniRakshak CLI")
    sub = p.add_subparsers(dest="cmd")

    sp = sub.add_parser("demo", help="run an offline end-to-end demo (no server, no audio)")
    sp.set_defaults(func=cmd_demo)

    sp = sub.add_parser("serve", help="run the FastAPI server")
    sp.add_argument("--host", default=None)
    sp.add_argument("--port", type=int, default=None)
    sp.add_argument("--reload", action="store_true")
    sp.set_defaults(func=cmd_serve)

    sp = sub.add_parser("eval", help="synthetic ASVspoof 5-style minDCF evaluation")
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("bench", help="measure pipeline latency p50/p95")
    sp.set_defaults(func=cmd_bench)

    args = p.parse_args(argv)
    if not hasattr(args, "func"):
        return cmd_serve(argparse.Namespace(host=None, port=None, reload=False))
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
