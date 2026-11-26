"""Export a static, backend-free snapshot of the web UI.

All API responses for a broad set of questions are precomputed into JSON files and
a small fetch() shim in the page serves them, so the demo can be hosted as plain
static files (e.g. GitHub Pages, an intranet share, or a claude.ai artifact).
Free-text questions that were not precomputed are answered with the closest
precomputed question, and the page says so.

Run:  python -m ontology_agent.web.export_static [out_dir]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from .. import AS_OF, ROOT
from ..agent import offline
from ..experiments.run import main as run_experiments
from ..kg.build import load_graph
from ..ontology import summary as ontology_summary
from ..ontology import to_turtle
from .server import EXAMPLES, STATIC, focus_subgraph, node_detail

GRAPH_TYPES = {"ClientGroup", "LegalEntity", "ExternalCompany", "Contact", "Banker", "CorporateEvent", "Deal",
               "Opportunity", "CallReport", "CreditPaper", "Product", "Intent"}
NODE_DETAIL_SKIP = {"Interaction", "WalletEstimate"}

SHIM = r"""<script>
/* Static snapshot: serve the demo's /api/* calls from precomputed JSON files. */
{
  const cache = {};
  const J = name => cache[name] || (cache[name] = window.__realFetch("static/" + name).then(r => r.json()));
  window.__realFetch = window.fetch.bind(window);
  const norm = s => String(s || "").toLowerCase().replace(/[’']/g, "'").replace(/\s+/g, " ").trim();
  const toks = s => new Set(norm(s).split(/[^a-z0-9一-鿿]+/).filter(t => t.length > 1));
  const ok = d => new Response(JSON.stringify(d), {headers: {"Content-Type": "application/json"}});
  window.fetch = async (url, opts = {}) => {
    const u = new URL(url, "https://x.invalid/"), p = u.pathname, q = u.searchParams;
    if (!p.startsWith("/api/")) return window.__realFetch(url, opts);
    if (p === "/api/meta") return ok(await J("meta.json"));
    if (p === "/api/ontology") return ok(await J("ontology.json"));
    if (p === "/api/experiments") return ok(await J("experiments.json"));
    if (p === "/api/node") { const n = await J("nodes.json"); return ok(n[q.get("id")] || {error: "not in snapshot"}); }
    if (p === "/api/graph") {
      const id = q.get("center"), d2 = q.get("depth") === "2" ? (await J("graph_d2.json"))[id] : null;
      const g = d2 || (await J("graph_d1.json"))[id];
      return ok(g || {nodes: [{id, label: id, type: "Node", focus: true}], edges: []});
    }
    if (p === "/api/feedback") {
      try { const k = "feedback"; const a = JSON.parse(localStorage.getItem(k) || "[]"); a.push(JSON.parse(opts.body)); localStorage.setItem(k, JSON.stringify(a)); } catch (e) {}
      return ok({ok: true});
    }
    if (p === "/api/ask") {
      const body = JSON.parse(opts.body || "{}"), A = await J("answers.json"), key = norm(body.question);
      let res = A[key + "||" + body.banker] || A[key];
      if (!res) {
        const t = toks(body.question); let best = null, bs = 0;
        for (const k of Object.keys(A)) { const kt = toks(k.split("||")[0]); let i = 0; t.forEach(x => kt.has(x) && i++);
          const s = i / (t.size + kt.size - i || 1); if (s > bs) { bs = s; best = k; } }
        res = best ? {...A[best]} : {answer: "No precomputed answer.", steps: [], route: "DOC_SEARCH", technology: "", linked_entities: []};
        res.answer = "_Static snapshot: this exact question was not precomputed, so the closest one is shown: **" +
          (res.question || "") + "**. Run the app locally to ask free-form questions._\n\n" + res.answer;
      }
      return ok(res);
    }
    return ok({error: "not in snapshot"});
  };
}
</script>
"""


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().replace("’", "'")).strip()


def question_set() -> list[tuple[str, str | None]]:
    g, _ = load_graph()
    qs: list[tuple[str, str | None]] = [(q, None) for _r, q in EXAMPLES]
    rms = [b for b in g.of_type("Banker") if g.outs(b, "COVERS")]
    for b in rms:
        qs.append(("What changed across my portfolio and what should I do?", b))
    for gid in g.of_type("ClientGroup"):
        s = g.name(gid).removesuffix(" Group")
        qs += [(f"What opportunity is implied by {s}'s subsidiaries, flows, wallet and recent events?", None),
               (f"What products does {s} have?", None),
               (f"What's {s}'s revenue?", None),
               (f"What is {s}'s share of wallet?", None),
               (f"Find the latest call report for {s}", None),
               (f"Summarise the credit paper for {s}", None),
               (f"Which contacts at {s} have we not met in the last 6 months?", None),
               (f"Who has the strongest relationship with the CFO of {s}?", None),
               (f"Which legal entities of {s} hold cash management with us?", None),
               (f"What similar deals have we executed for related clients of {s}?", None),
               (f"{s} 有哪些交叉销售机会？", None),
               (f"{s} 的收入是多少？", None)]
    plants = json.loads((ROOT / "data" / "ground_truth" / "relationship_plants.json").read_text())
    for p in plants:
        s = p["group_name"].removesuffix(" Group")
        qs.append((f"How are {s} and {p['company_b']} connected?", None))
        qs.append((f"{s} 和 {p['company_b']} 有什么关联？", None))
    for eid in g.of_type("CorporateEvent"):
        qs.append((f"What opportunities does {eid} imply?", None))
    return list(dict.fromkeys(qs))


def export(out: Path) -> dict:
    g, rep = load_graph()
    out.mkdir(parents=True, exist_ok=True)
    st = out / "static"
    st.mkdir(exist_ok=True)

    def dump(name, obj):
        (st / name).write_text(json.dumps(obj, default=str, separators=(",", ":")))

    answers, questions = {}, []
    for q, banker in question_set():
        res = offline.answer(q, banker or "BK01")
        res["subgraph"] = focus_subgraph(res.get("highlight", []))
        res["elapsed_ms"] = 0
        res["mode"] = "offline (static snapshot)"
        res.pop("data", None)
        for s in res["steps"]:
            s["result_preview"] = s["result_preview"][:400]
        answers[_norm(q) + (f"||{banker}" if banker else "")] = res
        questions.append(q)

    from ..agent.router import ROUTES
    meta = {"as_of": AS_OF, "llm_available": False, "static": True,
            "model": "", "routes": ROUTES, "examples": EXAMPLES, "graph": rep["graph"],
            "questions": sorted(set(questions)),
            "bankers": [{"id": b, "name": g.name(b), "role": g.node(b)["props"]["role"]} for b in g.of_type("Banker")],
            "clients": [{"id": c, "name": g.name(c)} for c in g.of_type("ClientGroup")]}
    dump("meta.json", meta)
    dump("answers.json", answers)
    d1 = {}
    for n in g.nodes:
        if g.type(n) in GRAPH_TYPES:
            sub = g.subgraph(n, depth=1, limit=60)
            for x in sub["nodes"]:
                x["focus"] = x["id"] == n
            d1[n] = sub
    dump("graph_d1.json", d1)
    d2 = {}
    for c in g.of_type("ClientGroup"):
        sub = g.subgraph(c, depth=2, limit=110)
        for x in sub["nodes"]:
            x["focus"] = x["id"] == c
        d2[c] = sub
    dump("graph_d2.json", d2)
    dump("nodes.json", {n: node_detail(n) for n in g.nodes if g.type(n) not in NODE_DETAIL_SKIP})
    dump("ontology.json", {**ontology_summary(), "turtle": to_turtle()})
    exp_path = ROOT / "reports" / "experiment_results.json"
    if not exp_path.exists():
        run_experiments()
    dump("experiments.json", json.loads(exp_path.read_text())
         | {"markdown": (ROOT / "reports" / "experiment_results.md").read_text()})
    html = (STATIC / "index.html").read_text()
    html = html.replace("<script>\nconst COLORS", SHIM + "<script>\nconst COLORS", 1)
    (out / "index.html").write_text(html)
    sizes = {p.name: p.stat().st_size for p in sorted(st.iterdir())}
    return {"questions": len(answers), "files": sizes}


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dist" / "static_demo"
    print(json.dumps(export(target), indent=2))
