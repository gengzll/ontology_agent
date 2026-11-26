"""Deterministic planner: route -> tool plan -> templated answer.

Runs with no LLM at all, so the demo (and the experiments) are reproducible.
The Claude-powered agent (llm.py) uses the *same* tools but plans freely and
writes the final narrative.
"""

from __future__ import annotations

import re
import time

from .. import AS_OF
from ..engines.analytics import pname
from ..ontology import PRODUCT_BY_CODE
from . import tools as T
from .router import ROUTES, Routing, route

PRODUCT_KEYWORDS = [
    (r"cash management|operating account|现金管理", "CASH_MGMT"), (r"cash pool|liquidity|流动性", "LIQUIDITY"),
    (r"virtual account", "VIRTUAL_ACCT"), (r"trade finance|letters? of credit|贸易融资", "TRADE_FIN"),
    (r"supply chain finance|\bscf\b|供应链金融", "SCF"), (r"\bfx\b|foreign exchange|外汇", "FX_HEDGE"),
    (r"interest rate swap|\birs\b|利率", "IRS_HEDGE"), (r"commodit", "COMMODITY_HEDGE"), (r"custody", "CUSTODY"),
    (r"revolving|\brcf\b", "RCF"), (r"term loan|project finance", "TERM_LOAN"), (r"\bbonds?\b|\bdcm\b", "DCM"),
    (r"\bipo\b|\becm\b|equity", "ECM"), (r"m&a|advisory|并购", "MNA"), (r"acquisition finance", "ACQ_FIN"),
]


def product_in(text: str) -> str | None:
    for pat, code in PRODUCT_KEYWORDS:
        if re.search(pat, text, re.I):
            return code
    return None


class Trace:
    def __init__(self):
        self.steps: list[dict] = []

    def call(self, tool: str, **args):
        t0 = time.perf_counter()
        try:
            res = T.call_tool(tool, args)
            err = None
        except Exception as e:  # surfaced in trace, never swallowed silently
            res, err = {"error": str(e)}, str(e)
        self.steps.append({"tool": tool, "args": args, "ms": round((time.perf_counter() - t0) * 1000, 1),
                           "error": err, "result_preview": _preview(res)})
        if err or (isinstance(res, dict) and "error" in res):
            raise ClarifyError(f"tool `{tool}` failed: {err or res['error']}")
        return res


def _preview(res: dict) -> str:
    import json
    s = json.dumps(res, default=str)
    return s[:600] + ("..." if len(s) > 600 else "")


def _usd(x: float) -> str:
    return f"USD {x/1e6:,.1f}m" if abs(x) >= 1e6 else f"USD {x:,.0f}"


def answer(question: str, persona_banker: str = "BK01") -> dict:
    r = route(question)
    tr = Trace()
    handler = HANDLERS.get(r.route, h_opportunity)
    try:
        md, items, data = handler(question, r, tr, persona_banker)
    except ClarifyError as e:
        md, items, data = f"**I need a bit more detail:** {e}", [], {}
    highlight = list(dict.fromkeys([e["id"] for e in r.entities] + T.node_ids_in(data)[:40]))
    return {"question": question, "route": r.route, "technology": r.technology, "router_scores": r.scores,
            "linked_entities": [{k: v for k, v in e.items() if k != "pos"} for e in r.entities],
            "mode": "offline", "steps": tr.steps, "answer": md, "answer_items": items, "highlight": highlight,
            "data": data}


class ClarifyError(Exception):
    pass


def _client(r: Routing) -> dict:
    c = r.first("ClientGroup", "LegalEntity")
    if not c:
        raise ClarifyError("which client? (e.g. 'Aurora Logistics')")
    return c


# ---------------------------------------------------------------------------
def h_doc_search(q, r, tr, _p):
    c = r.first("ClientGroup", "LegalEntity")
    doc_type = "CreditPaper" if re.search(r"credit paper|credit review|授信", q, re.I) else \
        "CallReport" if re.search(r"call report|meeting|拜访", q, re.I) else ""
    latest = bool(re.search(r"latest|most recent|last|最新|最近", q, re.I))
    res = tr.call("search_documents", query=q, client=c["id"] if c else "", doc_type=doc_type, latest=latest,
                  k=3 if latest else 5)
    hits = res.get("hits", [])
    if not hits:
        return "No matching documents found.", [], res
    md = [f"**{'Latest' if latest else 'Top'} {doc_type or 'documents'}"
          f"{' for ' + c['name'] if c else ''}** (vector search, {'sorted by date' if latest else 'by relevance'}):",
          ""]
    for h in hits:
        md.append(f"- **{h['title']}** `[{h['doc_id']}]` - {h['date']}  \n  _{h['snippet'][:260]}..._")
    return "\n".join(md), [h["doc_id"] for h in hits], res


def h_doc_summary(q, r, tr, _p):
    doc = r.first("CreditPaper", "CallReport")
    if not doc:
        c = _client(r)
        dt = "CallReport" if re.search(r"call report|拜访", q, re.I) else "CreditPaper"
        s = tr.call("search_documents", query=q, client=c["id"], doc_type=dt, latest=True, k=1)
        if not s.get("hits"):
            raise ClarifyError("I could not find that document.")
        doc_id = s["hits"][0]["doc_id"]
    else:
        doc_id = doc["id"]
    d = tr.call("get_document", doc_id=doc_id)
    md = [f"**Summary of {d['title']}** `[{doc_id}]` (author {d['author']}, {d['date']})", ""]
    md += [f"- {s}" for s in d["extractive_summary"]]
    if d["extracted_intents"]:
        md += ["", "**Signals extracted into the graph:** " + ", ".join(
            f"{i['intent']} ({i['polarity']})" for i in d["extracted_intents"])]
    md += ["", "_Offline mode uses an extractive summary; with an API key Claude writes an abstractive summary._"]
    return "\n".join(md), [doc_id], d


def h_metric(q, r, tr, _p):
    c = _client(r)
    res = tr.call("query_metric", client=c["id"], metric=r.metric or "revenue")
    if "error" in res:
        return res["error"], [], res
    rows = res["rows"]
    md = [f"**{res['metric'].replace('_', ' ').title()} - {res['client']}** ({res['definition']}, {res['unit']})", ""]
    keys = [k for k in res["columns"]]
    md.append("| " + " | ".join(keys) + " |")
    md.append("|" + "---|" * len(keys))
    for row in rows[-12:]:
        md.append("| " + " | ".join(f"{row[k]:,}" if isinstance(row[k], (int, float)) and k != "year" else str(row[k])
                                    for k in keys) + " |")
    md += ["", f"Governed SQL: `{res['sql']}`"]
    return "\n".join(md), [str(rows[-1]["value"])] if rows else [], res


def h_holdings(q, r, tr, _p):
    c = _client(r)
    res = tr.call("get_product_holdings", client=c["id"])
    md = [f"**Products held by {res['client']}** (graph: ClientGroup <- LegalEntity -HOLDS-> Product)", ""]
    for fam, ps in res["by_family"].items():
        md.append(f"- **{fam}**: {', '.join(ps) if ps else '-'}")
    md += ["", "| Legal entity | Country | Product | Since |", "|---|---|---|---|"]
    for h in res["holdings"]:
        md.append(f"| {h['entity']} | {h['country']} | {h['product']} | {h['since']} |")
    return "\n".join(md), sorted({h["product_code"] for h in res["holdings"]}), res


def h_opportunity(q, r, tr, _p):
    c = _client(r)
    res = tr.call("cross_sell_signals", client=c["id"], k=4)
    md = [f"### Implied opportunities - {res['client']}", "",
          f"Connected sources: subsidiaries ({len(res['subsidiaries'])} legal entities), payment flows, product "
          f"holdings, 2025 wallet, pipeline, call reports, events.", ""]
    for i, n in enumerate(res["next_best_actions"], 1):
        tag = " _(already in pipeline)_" if n["in_pipeline"] else " **(net-new)**"
        md.append(f"**{i}. {n['product']}** ({n['family']}) - score {n['score']:.2f}{tag}")
        md.append(f"   - sources: {', '.join(n['sources'])}")
        for ev in n["evidence"][:4]:
            md.append(f"   - {ev}")
    ev_items = []
    for ev in res["recent_events"]:
        if ev["date"] >= "2026-05-01":
            eo = tr.call("event_opportunities", event_id=ev["event_id"])
            prods = [x["product"] for x in eo["recommended"] if x["product"]]
            ev_items.append((ev, eo))
            md += ["", f"**Event-driven:** {ev['headline']} ({ev['date']}) `[{ev['event_id']}]` -> "
                       f"{', '.join(prods) or eo['recommended'][0]['rule']}"]
            for x in eo["rejected_by_context"]:
                md.append(f"   - not recommended: {x['product']} - {x['evidence']}")
    if res["suppressed_by_client_statement"]:
        md += ["", "**Suppressed** (client said no): " + "; ".join(
            f"{pname(s['product'])} - {s['evidence'][0]}" for s in res["suppressed_by_client_statement"])]
    w = res["wallet_2025"]
    md += ["", "Wallet 2025: " + ", ".join(f"{f} {_usd(v['wallet_usd'])} ({v['share']:.0%} share)"
                                           for f, v in w.items())]
    return "\n".join(md), [n["product_code"] for n in res["next_best_actions"]], {"cross_sell": res,
                                                                                 "events": [e[1] for e in ev_items]}


def h_connection(q, r, tr, _p):
    orgs = r.all("ClientGroup", "LegalEntity", "ExternalCompany")
    if len(orgs) < 2:
        raise ClarifyError("please name two organisations, e.g. 'How are Aurora Logistics and Harrowgate Steel "
                           "Trading connected?'")
    res = tr.call("find_connections", a=orgs[0]["id"], b=orgs[1]["id"], max_hops=3)
    conns = res.get("connections", [])
    if not conns:
        return f"No connection within 3 hops between {res.get('a')} and {res.get('b')}.", [], res
    direct = [c for c in conns if c["kind"] != "INDIRECT"]
    indirect = [c for c in conns if c["kind"] == "INDIRECT"]
    md = [f"**{res['a']} <-> {res['b']}: {len(direct)} direct and {len(indirect)} indirect connection(s)**", ""]
    label = {"JV": "Joint venture", "SUPPLIER": "Supplier payments", "SHARED_DIRECTOR": "Shared director",
             "FORMER_EMPLOYEE": "Former employee", "DEAL": "Deal with us", "EVENT": "Corporate event",
             "DOC_MENTION": "Mentioned in our documents", "TRADE": "Client-to-client trade", "EMPLOYEE": "Employee",
             "OTHER": "Other"}
    for c in direct:
        md.append(f"- **{label.get(c['kind'], c['kind'])}** ({c['hops']} hop{'s' if c['hops'] > 1 else ''}): "
                  f"`{c['explanation']}`")
    if indirect:
        md += ["", "Indirect (via third parties):"]
        for c in indirect[:5]:
            md.append(f"- via {c['via']}: `{c['explanation']}`")
    return "\n".join(md), [(c["kind"], c["via"]) for c in direct], res


def h_similar_deals(q, r, tr, _p):
    c = _client(r)
    prod = product_in(q)
    res = tr.call("related_parties_and_deals", client=c["id"], product=prod or "")
    cs = tr.call("cross_sell_signals", client=c["id"], k=3)
    focus = {n["product_code"] for n in cs["next_best_actions"]}
    deals = res["deals_for_related_parties"]
    wanted = [k for pat, k in ((r"\bjv|joint venture", "JV partner"), (r"supplier", "supplier"),
                               (r"trading partner|customer", "trading partner"), (r"peer", "industry peer"),
                               (r"people|former|director", "people link")) if re.search(pat, q, re.I)]
    if wanted:
        deals = [d for d in deals if any(d["related_via"].startswith(w) for w in wanted)]
    if re.search(r"subsidiar|own deals|子公司", q, re.I):
        deals = [{**d, "related_via": "own subsidiary", "related_party": d["client"]} for d in res["own_deals"]] + deals
    md = [f"**Deals executed for parties related to {res['client']}** (GraphRAG: graph expansion -> deal "
          f"retrieval)", ""]
    groups: dict[str, list] = {}
    for d in deals:
        groups.setdefault(d["related_via"], []).append(d)
    order = ["JV partner", "supplier", "trading partner", "people link", "industry peer"]
    for k in sorted(groups, key=lambda k: next((i for i, o in enumerate(order) if k.startswith(o)), 9)):
        md.append(f"- _{k}_")
        for d in groups[k][:4]:
            star = " **<- precedent for current opportunity**" if d["product_code"] in focus else ""
            md.append(f"  - `[{d['deal_id']}]` {d['date']} {d['product']} {_usd(d['size_usd'])} for {d['client']}"
                      f"{star}")
    if not deals:
        md.append("No deals found for related parties.")
    md += ["", f"Current top opportunities for {res['client']}: " +
           ", ".join(pname(p) for p in focus)]
    return "\n".join(md), [d["deal_id"] for d in deals], {"related": res, "focus_products": sorted(focus)}


def h_portfolio(q, r, tr, persona):
    b = r.first("Banker")
    res = tr.call("portfolio_changes", banker=b["id"] if b else persona, days=60)
    ch = res["changes"]
    md = [f"### Portfolio changes for {res['banker']} - last {res['window_days']} days (as of {AS_OF})",
          f"Clients: {', '.join(res['clients'])}", ""]
    icon = {3: "HIGH", 2: "MED", 1: "LOW"}
    for i, x in enumerate(ch[:15], 1):
        md.append(f"{i}. **[{icon[x['priority']]}] {x['client']}** - {x['what']}  \n   -> **Do:** {x['action']}")
        for ev in x["evidence"][:3]:
            md.append(f"   - {ev}")
    if not ch:
        md.append("No material changes detected.")
    return "\n".join(md), [x["client"] + ":" + x["kind"] for x in ch], res


def h_relationship(q, r, tr, _p):
    c = _client(r)
    m = re.search(r"(\d+)\s*(month|day)", q, re.I)
    days = (int(m.group(1)) * (30 if m.group(2).lower().startswith("month") else 1)) if m else 180
    res = tr.call("relationship_coverage", client=c["id"], days=days)
    prod = product_in(q)
    if re.search(r"not (met|seen)|haven'?t met|no (interaction|touchpoint)|没见", q, re.I):
        nm = res["not_met_in_window"]
        md = [f"**Contacts at {res['client']} with no touchpoint in the last {days} days** "
              f"(Contact -WORKS_AT-> LegalEntity -BELONGS_TO_GROUP-> Client; Interaction -WITH_CONTACT-> Contact)",
              ""]
        md += [f"- {x['name']} ({x['title']}, {x['entity']}) - last: {x['last_touchpoint'] or 'never'}"
               f"{' by ' + x['last_banker'] if x['last_banker'] else ''}" for x in nm] or ["- none"]
        return "\n".join(md), [x["name"] for x in nm], res
    if re.search(r"strongest|best relationship|who (knows|has met)|most (often|frequently)|关系最好", q, re.I):
        role = "CFO" if re.search(r"cfo|财务", q, re.I) else "Treasurer" if re.search(r"treasurer", q, re.I) else ""
        cand = [x for x in res["contacts"] if role and role in x["title"]] or res["contacts"]
        named = r.first("Contact")
        if named:
            cand = [x for x in res["contacts"] if x["contact_id"] == named["id"]] or cand
        x = cand[0]
        md = [f"**Bank relationships with {x['name']} ({x['title']}, {res['client']})**", ""]
        md += [f"- {b} - {n} touchpoints, last {d}" for b, n, d in x["bankers"]] or ["- no touchpoints"]
        if x["prior_employers"] or x["board_seats"]:
            md.append(f"- Background: prior employers {x['prior_employers'] or '-'}; boards {x['board_seats'] or '-'}")
        top = [x["bankers"][0][0]] if x["bankers"] else []
        return "\n".join(md), top, res
    if prod:
        ents = res["entities_by_product"].get(prod, [])
        md = [f"**Legal entities of {res['client']} holding {pname(prod)} with us**", ""]
        md += [f"- {e['entity']} ({e['country']}), since {e['since']} - coverage: {e['coverage_banker']}"
               for e in ents] or ["- none"]
        return "\n".join(md), [e["entity"] for e in ents], res
    md = [f"**Relationship map - {res['client']}**", "", "| Contact | Title | Touchpoints | Last | Top banker |",
          "|---|---|---|---|---|"]
    for x in res["contacts"]:
        md.append(f"| {x['name']} | {x['title']} | {x['touchpoints']} | {x['last_touchpoint']} | "
                  f"{x['bankers'][0][0] if x['bankers'] else '-'} |")
    return "\n".join(md), [x["name"] for x in res["contacts"]], res


def h_event(q, r, tr, _p):
    ev = r.first("CorporateEvent")
    if not ev:
        c = _client(r)
        evs = tr.call("list_events", client=c["id"])["events"]
        if not evs:
            raise ClarifyError("no events found for that client")
        eid = evs[0]["event_id"]
    else:
        eid = ev["id"]
    res = tr.call("event_opportunities", event_id=eid)
    e = res["event"]
    md = [f"**{e['headline']}** ({e['date']}) `[{eid}]`", "", "Recommended (ontology playbook x graph context as of "
                                                             "event date):"]
    for x in res["recommended"]:
        md.append(f"- **{x['product'] or 'Action'}** ({x['family']}) - {x['rule']}; _{x['evidence']}_")
    if res["rejected_by_context"]:
        md += ["", "Filtered out by client context:"]
        md += [f"- {x['product']} - {x['evidence']}" for x in res["rejected_by_context"]]
    for line in res["relationship_context"]:
        md.append(f"- Relationship: {line}")
    return "\n".join(md), [x["product_code"] for x in res["recommended"] if x["product_code"]], res


HANDLERS = {"DOC_SEARCH": h_doc_search, "DOC_SUMMARY": h_doc_summary, "METRIC": h_metric, "HOLDINGS": h_holdings,
            "OPPORTUNITY": h_opportunity, "CONNECTION": h_connection, "SIMILAR_DEALS": h_similar_deals,
            "PORTFOLIO": h_portfolio, "RELATIONSHIP": h_relationship, "EVENT": h_event}

assert set(HANDLERS) == set(ROUTES)
_ = PRODUCT_BY_CODE
