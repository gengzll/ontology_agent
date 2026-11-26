"""Command line entry point.

  python -m ontology_agent generate            # (re)create synthetic source systems
  python -m ontology_agent build               # build the knowledge graph and print quality stats
  python -m ontology_agent ask "question" [--mode offline|claude|auto] [--banker BK01] [--trace]
  python -m ontology_agent experiments         # run the 3 lighthouse experiments
  python -m ontology_agent serve [--port 8000] # web UI
  python -m ontology_agent export-ontology     # write ontology/cib_client_ontology.ttl
"""

from __future__ import annotations

import argparse
import json
import sys

from . import ROOT


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ontology_agent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate")
    sub.add_parser("build")
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--mode", default="auto", choices=["auto", "offline", "claude"])
    a.add_argument("--banker", default="BK01")
    a.add_argument("--trace", action="store_true")
    sub.add_parser("experiments")
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--host", default="127.0.0.1")
    sub.add_parser("export-ontology")
    args = ap.parse_args(argv)

    if args.cmd == "generate":
        from .synth.generate import main as gen
        gen()
    elif args.cmd == "build":
        from .kg.build import build_graph
        _g, rep = build_graph()
        print(json.dumps(rep, indent=2, default=str))
    elif args.cmd == "ask":
        from .agent.agent import ask
        res = ask(args.question, mode=args.mode, persona_banker=args.banker)
        print(f"[route] {res['route']}  ->  {res['technology']}   ({res['mode']})")
        if args.trace:
            for i, st in enumerate(res["steps"], 1):
                print(f"[tool {i}] {st['tool']}({json.dumps(st['args'])})  {st['ms']} ms")
        print()
        print(res["answer"])
    elif args.cmd == "experiments":
        from .experiments.run import main as run
        run()
    elif args.cmd == "serve":
        sys.argv = ["server", "--port", str(args.port), "--host", args.host]
        from .web.server import main as serve
        serve()
    elif args.cmd == "export-ontology":
        from .ontology import to_turtle
        out = ROOT / "ontology" / "cib_client_ontology.ttl"
        out.parent.mkdir(exist_ok=True)
        out.write_text(to_turtle())
        print("wrote", out)


if __name__ == "__main__":
    main()
