"""Question router: maps a frontline question to the best technology (the table
in the exploration brief) and links the entities it mentions to graph nodes."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from ..engines.semantic_sql import metric_from_text
from ..kg.build import load_graph

ROUTES: dict[str, dict] = {
    "DOC_SEARCH": {"technology": "Search / vector RAG", "example": "Find the latest call report"},
    "DOC_SUMMARY": {"technology": "LLM + RAG", "example": "Summarise this credit paper"},
    "METRIC": {"technology": "SQL / semantic layer", "example": "What's Client X's revenue?"},
    "HOLDINGS": {"technology": "SQL / graph", "example": "What products does Client X have?"},
    "OPPORTUNITY": {"technology": "Graph + analytics + LLM",
                    "example": "What opportunity is implied by Client X's subsidiaries, flows, wallet and recent events?"},
    "CONNECTION": {"technology": "Graph", "example": "How are Client A and Company B connected?"},
    "SIMILAR_DEALS": {"technology": "GraphRAG", "example": "What similar deals have we executed for related clients?"},
    "PORTFOLIO": {"technology": "Graph + analytics + agent",
                  "example": "What changed across my portfolio and what should I do?"},
    "RELATIONSHIP": {"technology": "Graph (multi-hop)",
                     "example": "Which contacts at Client X have we not met in 6 months?"},
    "EVENT": {"technology": "Graph + ontology playbook",
              "example": "What opportunities does event EV012 imply?"},
}

# (route, regex, weight) - English + Chinese cues
CUES = [
    ("PORTFOLIO", r"my portfolio|across my|my clients|my book|what changed|portfolio|组合|我的客户|变化", 3),
    ("SIMILAR_DEALS", r"similar deal|deals? have we (done|executed)|precedent|related client|deals? .*"
                      r"(subsidiar|jv|joint venture|supplier|related)|类似.*交易|相关客户", 3),
    ("CONNECTION", r"connected|connection|relationship between|link(ed)? between|how .* related|关联|联系|关系链", 3),
    ("EVENT", r"\bEV\d{3}\b|this event|the event|事件", 2),
    ("OPPORTUNITY", r"opportunit|cross.?sell|next best|what should (we|i) pitch|implied|upsell|机会|交叉销售|推荐", 2),
    ("DOC_SUMMARY", r"summari[sz]e|summary|tl;?dr|总结|摘要|概括", 3),
    ("DOC_SEARCH", r"\bfind\b|latest|most recent|search|look up|call report|credit paper|查找|最新|拜访报告|授信报告", 1),
    ("HOLDINGS", r"products?|holds?|holdings|产品|持有", 1),
    ("RELATIONSHIP", r"contacts?|not (met|seen)|haven'?t met|no (interaction|touchpoint)|who (knows|met)|"
                     r"strongest relationship|best relationship|which (legal )?entities|coverage|banker.*(met|know)|"
                     r"联系人|没见|覆盖|哪些实体", 2),
]

ID_RE = re.compile(r"\b(G\d{3}|E\d{4}|X\d{3}|C\d{4}|BK\d{2}|EV\d{3}|D\d{4}|CR\d{4}|CP\d{3}|OP\d{3})\b")


@dataclass
class Routing:
    route: str
    technology: str
    scores: dict
    entities: list[dict] = field(default_factory=list)
    metric: str | None = None

    def first(self, *types: str) -> dict | None:
        return next((e for e in self.entities if e["type"] in types), None)

    def all(self, *types: str) -> list[dict]:
        return [e for e in self.entities if e["type"] in types]


@lru_cache(maxsize=1)
def gazetteer() -> list[tuple[str, str, str]]:
    """(surface form lower, node id, type) sorted longest-first."""
    g, _ = load_graph()
    out = []
    first_tok_count: dict[str, int] = {}
    for gid in g.of_type("ClientGroup"):
        t = g.name(gid).split()[0].lower()
        first_tok_count[t] = first_tok_count.get(t, 0) + 1
    for gid in g.of_type("ClientGroup"):
        name = g.name(gid)
        short = name.removesuffix(" Group")
        out += [(name.lower(), gid, "ClientGroup"), (short.lower(), gid, "ClientGroup")]
        tok = name.split()[0].lower()
        if first_tok_count[tok] == 1:
            out.append((tok, gid, "ClientGroup"))
    for t in ("LegalEntity", "ExternalCompany", "Banker", "Contact"):
        for nid in g.of_type(t):
            if nid.startswith("UNRES:"):
                continue
            out.append((g.name(nid).lower(), nid, t))
            if t == "ExternalCompany":
                out.append((g.name(nid).split()[0].lower(), nid, t))
    out.sort(key=lambda x: -len(x[0]))
    return out


def link_entities(text: str) -> list[dict]:
    g, _ = load_graph()
    found, spans = [], []
    low = text.lower()
    for m in ID_RE.finditer(text):
        nid = m.group(1)
        if nid in g.nodes:
            found.append({"id": nid, "name": g.name(nid), "type": g.type(nid), "pos": m.start()})
            spans.append((m.start(), m.end()))
    for surface, nid, t in gazetteer():
        if len(surface) < 4:
            continue
        for m in re.finditer(r"(?<![a-z0-9])" + re.escape(surface) + r"(?![a-z0-9])", low):
            s, e = m.span()
            if any(not (e <= a or s >= b) for a, b in spans):
                continue
            spans.append((s, e))
            found.append({"id": nid, "name": g.name(nid), "type": t, "pos": s})
    found.sort(key=lambda x: x["pos"])
    uniq, seen = [], set()
    for f in found:
        if f["id"] not in seen:
            seen.add(f["id"])
            uniq.append(f)
    return uniq


def route(question: str) -> Routing:
    q = question.strip()
    scores = {r: 0.0 for r in ROUTES}
    for r, pat, w in CUES:
        if re.search(pat, q, re.I):
            scores[r] += w
    ents = link_entities(q)
    metric = metric_from_text(q)
    orgs = [e for e in ents if e["type"] in ("ClientGroup", "LegalEntity", "ExternalCompany")]
    if metric and not re.search(r"opportunit|机会|portfolio|组合", q, re.I):
        scores["METRIC"] += 2.5 if metric != "payment_volume" else 1.5
    if len(orgs) >= 2 and scores["CONNECTION"] > 0:
        scores["CONNECTION"] += 2
    if any(e["type"] == "CorporateEvent" for e in ents):
        scores["EVENT"] += 3
    if any(e["type"] in ("CallReport", "CreditPaper") for e in ents) and scores["DOC_SUMMARY"] == 0:
        scores["DOC_SEARCH"] += 1
    if scores["DOC_SUMMARY"] and scores["PORTFOLIO"] == 0:
        scores["DOC_SEARCH"] = 0
    if scores["RELATIONSHIP"] and scores["HOLDINGS"] and re.search(r"which (legal )?entities|哪些实体", q, re.I):
        scores["RELATIONSHIP"] += 1
    best = max(scores, key=lambda r: scores[r])
    if scores[best] == 0:
        best = "OPPORTUNITY" if orgs else "DOC_SEARCH"
    return Routing(route=best, technology=ROUTES[best]["technology"], scores=scores, entities=ents, metric=metric)
