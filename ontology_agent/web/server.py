"""Dependency-free web UI for the ontology agent demo.

Run:  python -m ontology_agent.web.server  [--port 8000]
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .. import AS_OF, DATA_DIR, REPORT_DIR
from ..agent import agent, llm
from ..agent.router import ROUTES
from ..kg.build import load_graph
from ..ontology import summary as ontology_summary
from ..ontology import to_turtle

STATIC = Path(__file__).parent / "static"

EXAMPLES = [
    ("DOC_SEARCH", "Find the latest call report for Aurora Logistics"),
    ("DOC_SUMMARY", "Summarise the credit paper for Banyan Foods"),
    ("METRIC", "What's Cobalt Semicon's revenue?"),
    ("HOLDINGS", "What products does Evergreen Textiles have?"),
    ("OPPORTUNITY", "What opportunity is implied by Aurora Logistics' subsidiaries, flows, wallet and recent events?"),
    ("CONNECTION", "How are Aurora Logistics and Harrowgate Steel Trading connected?"),
    ("SIMILAR_DEALS", "What similar deals have we executed for related clients of Banyan Foods?"),
    ("PORTFOLIO", "What changed across my portfolio and what should I do?"),
    ("RELATIONSHIP", "Which contacts at Jadeport Maritime have we not met in the last 6 months?"),
    ("RELATIONSHIP", "Who has the strongest relationship with the CFO of Kestrel Auto Components?"),
    ("EVENT", "What opportunities does EV043 imply?"),
    ("OPPORTUNITY", "Kestrel Auto Components 有哪些交叉销售机会？"),
]


def focus_subgraph(ids: list[str], limit: int = 45) -> dict:
    g, _ = load_graph()
    ids = [i for i in dict.fromkeys(ids) if i in g.nodes][:limit]
    s = set(ids)
    # pull in the parent group of any entity so fragments connect
    for i in list(ids):
        if g.type(i) == "LegalEntity":
            s.add(g.outs(i, "BELONGS_TO_GROUP")[0].dst)
    edges = {}
    for n in s:
        for e in g.outs(n):
            if e.dst in s:
                edges[(e.src, e.rel, e.dst)] = {"source": e.src, "target": e.dst, "rel": e.rel}
    return {"nodes": [{"id": n, "label": g.name(n), "type": g.type(n), "focus": n in ids} for n in s],
            "edges": list(edges.values())}


def node_detail(nid: str) -> dict:
    g, _ = load_graph()
    if nid not in g.nodes:
        return {"error": "unknown node"}
    n = g.node(nid)
    props = {k: v for k, v in n["props"].items() if k not in ("monthly_usd",)}
    rels = {}
    for e, nb, d in g.neighbours(nid):
        key = f"{e.rel} ({d})"
        rels.setdefault(key, []).append({"id": nb, "name": g.name(nb), "type": g.type(nb)})
    return {"id": nid, "type": n["type"], "props": props,
            "relations": {k: v[:25] for k, v in sorted(rels.items())}}


class Handler(BaseHTTPRequestHandler):
    server_version = "OntologyAgentDemo/1.0"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code: int, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else (json.dumps(body, default=str).encode()
                                                     if ctype == "application/json" else body.encode())
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        g, rep = load_graph()
        if u.path in ("/", "/index.html"):
            return self._send(200, (STATIC / "index.html").read_bytes(), "text/html")
        if u.path == "/api/meta":
            return self._send(200, {
                "as_of": AS_OF, "llm_available": llm.llm_available(), "model": llm.MODEL,
                "routes": ROUTES, "examples": EXAMPLES, "graph": rep["graph"], "er": rep["entity_resolution"],
                "bankers": [{"id": b, "name": g.name(b), "role": g.node(b)["props"]["role"]}
                            for b in g.of_type("Banker")],
                "clients": [{"id": c, "name": g.name(c)} for c in g.of_type("ClientGroup")]})
        if u.path == "/api/graph":
            center = q.get("center", "G001")
            if center not in g.nodes:
                return self._send(404, {"error": "unknown node"})
            sub = g.subgraph(center, depth=min(int(q.get("depth", 1)), 2), limit=int(q.get("limit", 80)))
            for n in sub["nodes"]:
                n["focus"] = n["id"] == center
            return self._send(200, sub)
        if u.path == "/api/node":
            return self._send(200, node_detail(q.get("id", "")))
        if u.path == "/api/ontology":
            return self._send(200, {**ontology_summary(), "turtle": to_turtle()})
        if u.path == "/api/experiments":
            path = REPORT_DIR / "experiment_results.json"
            if not path.exists():
                from ..experiments.run import main as run_all
                run_all()
            return self._send(200, json.loads(path.read_text()) | {
                "markdown": (REPORT_DIR / "experiment_results.md").read_text()})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, {"error": "invalid JSON"})
        if u.path == "/api/ask":
            qtext = (body.get("question") or "").strip()
            if not qtext:
                return self._send(400, {"error": "empty question"})
            t0 = time.perf_counter()
            res = agent.ask(qtext, mode=body.get("mode", "auto"), persona_banker=body.get("banker", "BK01"))
            res["elapsed_ms"] = round((time.perf_counter() - t0) * 1000)
            res["subgraph"] = focus_subgraph(res.get("highlight", []))
            res.pop("data", None)
            return self._send(200, res)
        if u.path == "/api/feedback":
            rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "question": body.get("question"),
                   "route": body.get("route"), "verdict": body.get("verdict"), "comment": body.get("comment", "")}
            with open(DATA_DIR / "feedback.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            return self._send(200, {"ok": True})
        return self._send(404, {"error": "not found"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    load_graph()
    print(f"Ontology agent demo on http://{a.host}:{a.port}  (LLM: "
          f"{'Claude ' + llm.MODEL if llm.llm_available() else 'not configured - offline planner'})")
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
