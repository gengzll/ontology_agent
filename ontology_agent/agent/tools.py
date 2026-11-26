"""Tool registry shared by the LLM agent and the offline planner.

Each tool is a plain function returning JSON-serialisable data plus a JSON schema
so that it can be offered to Claude as a client tool.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from .. import AS_OF
from ..engines import analytics as A
from ..engines import semantic_sql as S
from ..engines.vector import index, summarise_extractive
from ..kg.build import load_graph
from ..ontology import summary as ontology_summary
from .router import link_entities

NODE_ID_RE = re.compile(r"\b(G\d{3}|E\d{4}|X\d{3}|C\d{4}|BK\d{2}|EV\d{3}|D\d{4}|CR\d{4}|CP\d{3}|OP\d{3})\b")


def G():
    return load_graph()[0]


def resolve(ref: str, types: tuple[str, ...] = ("ClientGroup",)) -> str | None:
    g = G()
    ref = (ref or "").strip()
    if ref in g.nodes:
        nid = ref
    else:
        cands = [e for e in link_entities(ref) if e["type"] in types or not types]
        if not cands:
            cands = [e for e in link_entities(ref)]
        if not cands:
            return None
        nid = cands[0]["id"]
    if "ClientGroup" in types and g.type(nid) == "LegalEntity" and "LegalEntity" not in types:
        nid = g.outs(nid, "BELONGS_TO_GROUP")[0].dst
    return nid


def _need(ref, types=("ClientGroup",)):
    nid = resolve(ref, types)
    if not nid:
        raise ValueError(f"Could not resolve '{ref}' to a {'/'.join(types)} in the knowledge graph.")
    return nid


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------
def t_resolve_entity(name: str) -> dict:
    return {"matches": [{k: v for k, v in e.items() if k != "pos"} for e in link_entities(name)][:8]}


def t_client_profile(client: str) -> dict:
    return A.client_profile(G(), _need(client))


def t_product_holdings(client: str) -> dict:
    gid = _need(client)
    hs = A.holdings(G(), gid)
    sql = S.query_metric("product_holdings", gid)
    return {"client": G().name(gid), "group_id": gid, "holdings": hs, "sql": sql.get("sql"),
            "by_family": {f: sorted({h["product"] for h in hs if h["family"] == f}) for f in ("GB", "TB", "Markets")}}


def t_query_metric(client: str, metric: str) -> dict:
    gid = _need(client)
    res = S.query_metric(metric, gid)
    res["client"] = G().name(gid)
    res["group_id"] = gid
    return res


def t_run_sql(sql: str) -> dict:
    return S.run_sql(sql)


def t_sql_schema() -> dict:
    return {"tables": S.schema(), "metrics": {k: v["desc"] for k, v in S.METRICS.items()}}


def t_search_documents(query: str, client: str = "", doc_type: str = "", latest: bool = False, k: int = 5) -> dict:
    gid = resolve(client) if client else None
    hits = index().search(query, k=k, group_id=gid, doc_type=doc_type or None, sort_by_date=latest)
    for h in hits:
        h["snippet"] = h.pop("text")[:400]
    return {"query": query, "filter_client": G().name(gid) if gid else None, "hits": hits}


def t_get_document(doc_id: str) -> dict:
    g = G()
    if doc_id not in g.nodes or g.type(doc_id) not in ("CallReport", "CreditPaper"):
        raise ValueError(f"Unknown document {doc_id}")
    p = g.node(doc_id)["props"]
    return {"doc_id": doc_id, "type": g.type(doc_id), "title": p["title"], "date": p["date"], "text": p["text"],
            "about": g.name(p["group_id"]), "author": g.name(g.outs(doc_id, "AUTHORED_BY")[0].dst),
            "attendees": [g.name(e.dst) for e in g.outs(doc_id, "ATTENDEE")],
            "extracted_intents": [{"intent": g.name(e.dst), **e.props} for e in g.outs(doc_id, "EXPRESSES")],
            "extractive_summary": summarise_extractive(p["text"])}


def t_cross_sell(client: str, k: int = 3) -> dict:
    g = G()
    gid = _need(client)
    recent = [A.event_record(g, e.src) for e in g.ins(gid, "AFFECTS")]
    recent = sorted(recent, key=lambda x: x["date"], reverse=True)[:3]
    return {"client": g.name(gid), "group_id": gid,
            "next_best_actions": A.next_best_actions(g, gid, k),
            "all_signals": A.client_signals(g, gid),
            "suppressed_by_client_statement": A.suppressed_products(g, gid),
            "recent_events": recent,
            "subsidiaries": [e["name"] + f" ({e['country']})" for e in A.client_profile(g, gid)["legal_entities"]],
            "wallet_2025": A.client_profile(g, gid)["wallet_2025"]}


def t_list_events(client: str = "", since: str = "") -> dict:
    g = G()
    gid = resolve(client) if client else None
    evs = [A.event_record(g, e) for e in g.of_type("CorporateEvent")]
    evs = [e for e in evs if (not gid or e["group_id"] == gid) and (not since or e["date"] >= since)]
    return {"events": sorted(evs, key=lambda x: x["date"], reverse=True)[:30]}


def t_event_opportunities(event_id: str) -> dict:
    return A.event_opportunities(G(), event_id)


def t_find_connections(a: str, b: str, max_hops: int = 3) -> dict:
    types = ("ClientGroup", "LegalEntity", "ExternalCompany")
    na, nb = _need(a, types), _need(b, types)
    return A.find_connections(G(), na, nb, max_hops)


def t_related_deals(client: str, product: str = "") -> dict:
    g = G()
    gid = _need(client)
    rel = A.related_parties(g, gid)
    deals = []
    for r in rel:
        for d in A.deals_for_org(g, r["id"]):
            if product and product.upper() not in (d["product_code"], d["product"].upper()):
                continue
            deals.append({**d, "related_via": f"{r['relation']}: {r['why']}", "related_party": r["name"]})
    own = A.deals_for_org(g, gid)
    deals.sort(key=lambda d: d["date"], reverse=True)
    return {"client": g.name(gid), "group_id": gid, "related_parties": rel, "deals_for_related_parties": deals[:60],
            "own_deals": sorted(own, key=lambda d: d["date"], reverse=True)[:10]}


def t_portfolio_changes(banker: str, days: int = 60) -> dict:
    return A.portfolio_changes(G(), _need(banker, ("Banker",)), days)


def t_relationship_coverage(client: str, days: int = 180) -> dict:
    g = G()
    gid = _need(client)
    cs = A.contacts(g, gid)
    out = []
    for c in cs:
        tps = A.contact_touchpoints(g, c["contact_id"])
        out.append({**c, "touchpoints": len(tps), "last_touchpoint": tps[0]["date"] if tps else None,
                    "bankers": A.strongest_bankers_for_contact(g, c["contact_id"]),
                    "prior_employers": [g.name(e.dst) for e in g.outs(c["contact_id"], "PREVIOUSLY_WORKED_AT")],
                    "board_seats": [g.name(e.dst) for e in g.outs(c["contact_id"], "DIRECTOR_OF")]})
    return {"client": g.name(gid), "group_id": gid, "as_of": AS_OF, "contacts": out,
            "not_met_in_window": A.uncovered_contacts(g, gid, days),
            "entities_by_product": {p: A.entities_with_product(g, gid, p)
                                    for p in sorted(A.held_codes(g, gid))}}


def t_graph_neighbourhood(node_id: str, depth: int = 1) -> dict:
    g = G()
    nid = node_id if node_id in g.nodes else _need(node_id, ())
    sub = g.subgraph(nid, depth=min(depth, 2), limit=60)
    return {"center": nid, "name": g.name(nid), "type": g.type(nid), "props": {
        k: v for k, v in g.node(nid)["props"].items() if k not in ("text", "financials")}, **sub}


def t_ontology_schema() -> dict:
    s = ontology_summary()
    return {"classes": s["classes"], "relations": s["relations"], "products": s["products"],
            "event_playbook": s["event_playbook"], "signal_rules": s["signal_rules"]}


# ---------------------------------------------------------------------------
# Registry with JSON schemas (for Claude tool use)
# ---------------------------------------------------------------------------
def _schema(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


CLIENT = {"type": "string", "description": "Client group name or id, e.g. 'Aurora Logistics' or 'G001'."}

TOOLS: dict[str, dict[str, Any]] = {
    "resolve_entity": {"fn": t_resolve_entity, "desc": "Link a free-text name to knowledge-graph nodes (clients, "
                       "legal entities, external companies, bankers, contacts).",
                       "schema": _schema({"name": {"type": "string"}}, ["name"])},
    "get_client_profile": {"fn": t_client_profile, "desc": "Client group profile: legal-entity tree with ownership, "
                           "coverage team, financials, rating, 2025 wallet.", "schema": _schema({"client": CLIENT},
                                                                                              ["client"])},
    "get_product_holdings": {"fn": t_product_holdings, "desc": "Products held with the bank per legal entity "
                             "(SQL / graph).", "schema": _schema({"client": CLIENT}, ["client"])},
    "query_metric": {"fn": t_query_metric, "desc": "Governed semantic-layer metric for a client. Metrics: "
                     + ", ".join(S.METRICS), "schema": _schema({"client": CLIENT, "metric": {"type": "string",
                                                                                               "enum": list(S.METRICS)}},
                                                               ["client", "metric"])},
    "sql_schema": {"fn": t_sql_schema, "desc": "List SQL tables/columns and governed metric definitions.",
                   "schema": _schema({}, [])},
    "run_sql": {"fn": t_run_sql, "desc": "Run one read-only SQLite SELECT over source-system tables. Prefer "
                "query_metric for governed metrics.", "schema": _schema({"sql": {"type": "string"}}, ["sql"])},
    "search_documents": {"fn": t_search_documents, "desc": "Vector search over call reports, credit papers, deal "
                         "tombstones and news. Use latest=true to sort by date.",
                         "schema": _schema({"query": {"type": "string"}, "client": CLIENT,
                                            "doc_type": {"type": "string", "enum": ["", "CallReport", "CreditPaper",
                                                                                    "DealTombstone", "News"]},
                                            "latest": {"type": "boolean"}, "k": {"type": "integer"}}, ["query"])},
    "get_document": {"fn": t_get_document, "desc": "Full text + metadata + extracted intents of a CallReport "
                     "(CRxxxx) or CreditPaper (CPxxx).", "schema": _schema({"doc_id": {"type": "string"}},
                                                                          ["doc_id"])},
    "cross_sell_signals": {"fn": t_cross_sell, "desc": "Graph analytics for cross-sell / next best action: connects "
                           "payments, holdings, ownership, wallet, pipeline, call-report intents and peers; returns "
                           "ranked products with evidence, suppressed products, recent events.",
                           "schema": _schema({"client": CLIENT, "k": {"type": "integer"}}, ["client"])},
    "list_events": {"fn": t_list_events, "desc": "Corporate events (news) optionally filtered by client and date.",
                    "schema": _schema({"client": CLIENT, "since": {"type": "string",
                                                                   "description": "ISO date"}}, [])},
    "event_opportunities": {"fn": t_event_opportunities, "desc": "Evaluate the ontology event playbook for an event "
                            "against the graph as of the event date; returns recommended and context-rejected "
                            "GB/TB/Markets products with reasons.",
                            "schema": _schema({"event_id": {"type": "string"}}, ["event_id"])},
    "find_connections": {"fn": t_find_connections, "desc": "Multi-hop explainable paths between two organisations "
                         "(ownership, payments, JVs, shared directors, former employees, deals, documents).",
                         "schema": _schema({"a": {"type": "string"}, "b": {"type": "string"},
                                            "max_hops": {"type": "integer"}}, ["a", "b"])},
    "related_parties_and_deals": {"fn": t_related_deals, "desc": "GraphRAG: related parties of a client (JV "
                                  "partners, suppliers, trading partners, people links, industry peers) and the "
                                  "deals we executed for them.",
                                  "schema": _schema({"client": CLIENT, "product": {"type": "string"}}, ["client"])},
    "portfolio_changes": {"fn": t_portfolio_changes, "desc": "Changes across a banker's portfolio in the last N days "
                          "(events, flow anomalies, client intents, pipeline moves, coverage gaps) with suggested "
                          "actions.", "schema": _schema({"banker": {"type": "string", "description": "Banker name/id"},
                                                         "days": {"type": "integer"}}, ["banker"])},
    "relationship_coverage": {"fn": t_relationship_coverage, "desc": "Contacts at a client, touchpoints by banker, "
                              "contacts not met within N days, prior employers / board seats, entities by product.",
                              "schema": _schema({"client": CLIENT, "days": {"type": "integer"}}, ["client"])},
    "graph_neighbourhood": {"fn": t_graph_neighbourhood, "desc": "Raw neighbourhood of any node (for exploration).",
                            "schema": _schema({"node_id": {"type": "string"}, "depth": {"type": "integer"}},
                                              ["node_id"])},
    "ontology_schema": {"fn": t_ontology_schema, "desc": "The CIB ontology: classes, relations, product catalogue, "
                        "event playbook and signal rules.", "schema": _schema({}, [])},
}


def call_tool(name: str, args: dict) -> dict:
    spec = TOOLS.get(name)
    if not spec:
        raise ValueError(f"Unknown tool {name}")
    return spec["fn"](**args)


def claude_tool_defs() -> list[dict]:
    return [{"name": n, "description": t["desc"], "input_schema": t["schema"]} for n, t in TOOLS.items()]


def node_ids_in(obj: Any) -> list[str]:
    g = G()
    ids = NODE_ID_RE.findall(json.dumps(obj, default=str))
    return [i for i in dict.fromkeys(ids) if i in g.nodes]


ToolFn = Callable[..., dict]
