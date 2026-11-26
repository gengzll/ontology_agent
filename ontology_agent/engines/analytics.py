"""Graph analytics on top of the knowledge graph.

Every function returns *evidence* (graph facts with node ids) alongside the
result so that answers are explainable and auditable by bankers.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, timedelta

from .. import AS_OF
from ..kg.build import group_of
from ..kg.store import KnowledgeGraph
from ..ontology import (COMMODITY_INTENSIVE, COUNTRIES, EVENT_PLAYBOOK, INVESTMENT_GRADE, PRODUCT_BY_CODE,
                        SIGNAL_RULES, TRADE_INTENSIVE)

TODAY = date.fromisoformat(AS_OF)
RULE_W = {r["signal"]: r["weight"] for r in SIGNAL_RULES}
RULE_W.update({"FLOATING_DEBT_UNHEDGED": 0.15, "PIPELINE": 0.20})
EVENT_DRIVEN = {"MNA", "ACQ_FIN", "ECM"}
CREDIT_GATED = {"ACQ_FIN", "TERM_LOAN", "RCF"}


def pname(code: str) -> str:
    return PRODUCT_BY_CODE[code]["name"] if code in PRODUCT_BY_CODE else code


def _d(s: str) -> date:
    return date.fromisoformat(s)


# ---------------------------------------------------------------------------
# Basic lookups
# ---------------------------------------------------------------------------
def entities(g: KnowledgeGraph, gid: str) -> list[str]:
    return sorted(e.src for e in g.ins(gid, "BELONGS_TO_GROUP"))


def holdings(g: KnowledgeGraph, gid: str, as_of: date | None = None) -> list[dict]:
    out = []
    for e in entities(g, gid):
        for h in g.outs(e, "HOLDS"):
            if as_of is None or _d(h.props["since"]) <= as_of:
                code = h.dst.split(":", 1)[1]
                out.append({"entity_id": e, "entity": g.name(e), "country": g.node(e)["props"]["country"],
                            "product_code": code, "product": pname(code),
                            "family": PRODUCT_BY_CODE[code]["family"], "since": h.props["since"],
                            "limit_usd": h.props.get("limit_usd", 0)})
    return out


def held_codes(g, gid, as_of=None) -> set[str]:
    return {h["product_code"] for h in holdings(g, gid, as_of)}


def financials(g, gid) -> list[dict]:
    return g.node(gid)["props"].get("financials", [])


def leverage(g, gid, before: date | None = None) -> float:
    rows = financials(g, gid)
    if before:
        rows = [r for r in rows if r["year"] < before.year] or rows
    r = max(rows, key=lambda x: x["year"])
    return r["net_debt_usd_m"] / r["ebitda_usd_m"]


def wallet(g, gid, year: int = 2025) -> dict:
    out = {}
    for e in g.outs(gid, "HAS_WALLET"):
        p = g.node(e.dst)["props"]
        if p["year"] == year:
            out[p["family"]] = p
    return out


def client_profile(g, gid) -> dict:
    p = g.node(gid)["props"]
    ents = entities(g, gid)
    tree = []
    for e in ents:
        par = g.outs(e, "SUBSIDIARY_OF")
        tree.append({"entity_id": e, "name": g.name(e), "country": g.node(e)["props"]["country"],
                     "type": g.node(e)["props"]["entity_type"],
                     "parent": par[0].dst if par else None,
                     "ownership_pct": par[0].props.get("ownership_pct") if par else None})
    cov = [{"banker_id": e.src, "banker": g.name(e.src), "role": e.props["role"]} for e in g.ins(gid, "COVERS")]
    return {"group_id": gid, "name": p["name"], "industry": p["industry"], "hq_country": p["hq_country"],
            "functional_ccy": p["functional_ccy"], "risk_rating": p["risk_rating"], "segment": p["segment"],
            "financials": financials(g, gid), "legal_entities": tree, "coverage": cov,
            "wallet_2025": {f: {"wallet_usd": w["wallet_usd"], "share": w["share"]} for f, w in wallet(g, gid).items()}}


# ---------------------------------------------------------------------------
# Cross-sell / Next Best Action (Experiment 1)
# ---------------------------------------------------------------------------
def _flows(g, gid):
    for e in entities(g, gid):
        for p in g.outs(e, "PAYS"):
            yield e, p


def client_signals(g: KnowledgeGraph, gid: str, as_of: date = TODAY) -> list[dict]:
    """Detect cross-sell signals by connecting payments, holdings, wallet, pipeline and call reports."""
    prof = g.node(gid)["props"]
    fccy = prof["functional_ccy"]
    held = held_codes(g, gid)
    sig: list[dict] = []

    # 1. Unhedged FX flows (payments x holdings)
    fx, fx_nodes = defaultdict(int), []
    for e, p in _flows(g, gid):
        if p.props["purpose"] != "intercompany" and p.props["currency"] not in (fccy, "USD"):
            fx[p.props["currency"]] += p.props["annual_usd"]
            fx_nodes += [e, p.dst]
    tot = sum(fx.values())
    if tot > 10e6 and "FX_HEDGE" not in held:
        sig.append({"signal": "UNHEDGED_FX_FLOWS", "product": "FX_HEDGE", "strength": min(1, tot / 4e7),
                    "node_ids": list(dict.fromkeys(fx_nodes))[:8],
                    "evidence": [f"USD {v/1e6:,.1f}m/yr of third-party payments in {c} (functional ccy {fccy})"
                                 for c, v in sorted(fx.items(), key=lambda x: -x[1])[:3]]
                    + ["No FX hedging product held with us"]})

    # 2. Supplier concentration without SCF (payments x holdings)
    sup = defaultdict(int)
    for e, p in _flows(g, gid):
        if p.props["purpose"] == "supplier" and group_of(g, p.dst) != gid:
            sup[p.dst] += p.props["annual_usd"]
    big = {k: v for k, v in sup.items() if v >= 10e6}
    if len(big) >= 4 and "SCF" not in held:
        sig.append({"signal": "SUPPLIER_CONCENTRATION", "product": "SCF", "strength": min(1, sum(big.values()) / 1.2e8),
                    "node_ids": list(big)[:8],
                    "evidence": [f"{len(big)} suppliers >= USD 10m/yr, total USD {sum(big.values())/1e6:,.0f}m"]
                    + [f"Supplier {g.name(k)}: USD {v/1e6:,.1f}m/yr" for k, v in
                       sorted(big.items(), key=lambda x: -x[1])[:3]] + ["No Supply Chain Finance programme"]})

    # 3. Multi-country operating accounts without pooling (ownership graph x holdings)
    cm = holdings(g, gid)
    cm_countries = sorted({h["country"] for h in cm if h["product_code"] == "CASH_MGMT"})
    if len(cm_countries) >= 3 and "LIQUIDITY" not in held:
        sig.append({"signal": "MULTI_COUNTRY_NO_POOLING", "product": "LIQUIDITY",
                    "node_ids": sorted({h["entity_id"] for h in cm if h["product_code"] == "CASH_MGMT"}),
                    "strength": min(1, len(cm_countries) / 4),
                    "evidence": [f"Operating accounts with us in {len(cm_countries)} countries: "
                                 f"{', '.join(cm_countries)}", "No cash pooling / liquidity structure"]})

    # 4. Subsidiary banks elsewhere (payments rail x holdings x ownership)
    for e, p in _flows(g, gid):
        if (p.props["purpose"] == "intercompany" and p.props["rail"] == "OTHER_BANK"
                and g.type(p.dst) == "LegalEntity" and not any(h.dst == "P:CASH_MGMT" for h in g.outs(p.dst, "HOLDS"))):
            sig.append({"signal": "SUBSIDIARY_BANKS_ELSEWHERE", "product": "CASH_MGMT", "strength": 1.0,
                        "target_entity": p.dst, "node_ids": [e, p.dst],
                        "evidence": [f"{g.name(e)} funds {g.name(p.dst)} (USD {p.props['annual_usd']/1e6:,.1f}m/yr)"
                                     f" into an account at another bank", f"{g.name(p.dst)} has no operating "
                                                                            f"account with us"]})
            break

    # 5. Term debt without rates hedge (holdings)
    if "TERM_LOAN" in held and "IRS_HEDGE" not in held:
        sig.append({"signal": "FLOATING_DEBT_UNHEDGED", "product": "IRS_HEDGE", "strength": 0.6,
                    "evidence": ["Term loan with us", "No interest-rate hedge in place"]})

    # 6. Wallet gap (wallet x holdings x peers)
    popular = product_popularity(g)
    for c in EVENT_DRIVEN:
        popular.pop(c, None)  # deal products are triggered by events, not by peer / wallet gaps
    for fam, w in wallet(g, gid).items():
        if w["share"] < 0.10 and w["wallet_usd"] > 1e6:
            cands = [c for c, _ in popular.most_common() if PRODUCT_BY_CODE[c]["family"] == fam and c not in held]
            if cands:
                sig.append({"signal": "WALLET_GAP", "product": cands[0], "strength": 0.8,
                            "evidence": [f"{fam} wallet USD {w['wallet_usd']/1e6:,.1f}m with only "
                                         f"{w['share']:.0%} share of wallet"]})

    # 7. Intents from call reports (documents -> extracted Intent nodes)
    for d in g.ins(gid, "ABOUT"):
        if g.type(d.src) != "CallReport":
            continue
        dd = _d(g.node(d.src)["props"]["date"])
        if not (0 <= (as_of - dd).days <= 365):
            continue
        for x in g.outs(d.src, "EXPRESSES"):
            code = g.node(x.dst)["props"].get("product")
            if not code:
                continue
            sig.append({"signal": "CALL_REPORT_INTENT" if x.props["polarity"] == "positive" else "DECLINED_INTENT",
                        "product": code, "strength": 1.0 if x.props["polarity"] == "positive" else -1.0,
                        "doc_id": d.src,
                        "evidence": [f"Call report {d.src} ({dd}): \"{x.props['sentence']}\""]})

    # 8. Pipeline (CRM)
    for o in g.outs(gid, "HAS_OPPORTUNITY"):
        p = g.node(o.dst)["props"]
        if p["status"] == "Open":
            sig.append({"signal": "PIPELINE", "product": p["product_code"], "strength": 0.8, "opp_id": o.dst,
                        "evidence": [f"Open pipeline {o.dst} at stage {p['stage']}"]})

    # 9. Peer adoption
    for c, n in popular.items():
        if n / max(1, len(g.of_type("ClientGroup"))) >= 0.5 and c not in held:
            sig.append({"signal": "PEER_ADOPTION", "product": c, "strength": 0.5,
                        "evidence": [f"{n} of {len(g.of_type('ClientGroup'))} client groups hold {pname(c)}"]})
    return sig


def product_popularity(g) -> Counter:
    c = Counter()
    for gid in g.of_type("ClientGroup"):
        c.update(held_codes(g, gid))
    return c


def next_best_actions(g: KnowledgeGraph, gid: str, k: int = 3, sources: set[str] | None = None) -> list[dict]:
    """Rank products by noisy-OR of signal weights. `sources` restricts the data sources (for ablations)."""
    held = held_codes(g, gid)
    sigs = client_signals(g, gid)
    if sources is not None:
        sigs = [s for s in sigs if _source_of(s["signal"]) in sources]
    by_prod: dict[str, list[dict]] = defaultdict(list)
    for s in sigs:
        by_prod[s["product"]].append(s)
    recs = []
    for code, ss in by_prod.items():
        if code in held or code not in PRODUCT_BY_CODE:
            continue
        if any(s["strength"] < 0 for s in ss):
            continue  # client explicitly declined - suppress (and say so in the trace)
        if code in CREDIT_GATED and leverage(g, gid) >= 3.5:
            continue  # credit gate from the ontology: no new leverage above 3.5x
        miss = 1.0
        for s in ss:
            miss *= 1 - RULE_W.get(s["signal"], 0.1) * s["strength"]
        score = 1 - miss
        recs.append({"product_code": code, "product": pname(code), "family": PRODUCT_BY_CODE[code]["family"],
                     "score": round(score, 3), "signals": sorted({s["signal"] for s in ss}),
                     "sources": sorted({_source_of(s["signal"]) for s in ss}),
                     "in_pipeline": any(s["signal"] == "PIPELINE" for s in ss),
                     "node_ids": list(dict.fromkeys(n for s in ss for n in s.get("node_ids", [])
                                                    + [s.get("doc_id"), s.get("opp_id")] if n)),
                     "evidence": [ev for s in ss for ev in s["evidence"]]})
    recs.sort(key=lambda r: -r["score"])
    return recs[:k]


def suppressed_products(g, gid) -> list[dict]:
    out = [s for s in client_signals(g, gid) if s["strength"] < 0]
    lev = leverage(g, gid)
    if lev >= 3.5:
        out += [{"signal": "CREDIT_GATE", "product": c, "strength": -1,
                 "evidence": [f"leverage {lev:.1f}x >= 3.5x - no new leverage products"]}
                for c in sorted({s["product"] for s in client_signals(g, gid)} & CREDIT_GATED)]
    return out


def _source_of(sig: str) -> str:
    return {"UNHEDGED_FX_FLOWS": "payments", "SUPPLIER_CONCENTRATION": "payments",
            "MULTI_COUNTRY_NO_POOLING": "holdings+ownership", "SUBSIDIARY_BANKS_ELSEWHERE": "payments+ownership",
            "FLOATING_DEBT_UNHEDGED": "holdings", "WALLET_GAP": "wallet", "CALL_REPORT_INTENT": "call reports",
            "DECLINED_INTENT": "call reports", "PIPELINE": "pipeline", "PEER_ADOPTION": "peers"}.get(sig, sig)


def baseline_nba(g: KnowledgeGraph, gid: str, k: int = 3) -> list[str]:
    """'Today's' opportunity identification: CRM pipeline + product-gap vs. what most clients hold."""
    held = held_codes(g, gid)
    stage_rank = {"Mandated": 0, "Proposal": 1, "Qualified": 2, "Identified": 3}
    pipe = sorted([g.node(o.dst)["props"] for o in g.outs(gid, "HAS_OPPORTUNITY")],
                  key=lambda p: stage_rank.get(p["stage"], 9))
    out = [p["product_code"] for p in pipe if p["product_code"] not in held]
    for c, _ in product_popularity(g).most_common():
        if c not in held and c not in out:
            out.append(c)
    return list(dict.fromkeys(out))[:k]


# ---------------------------------------------------------------------------
# Event -> Opportunity (Experiment 2)
# ---------------------------------------------------------------------------
def _cond(g: KnowledgeGraph, name: str, gid: str, ev: dict, at: date) -> tuple[bool, str]:
    p = g.node(gid)["props"]
    cc = ev.get("country", "")
    held = held_codes(g, gid, at)
    if name == "always":
        return True, "applies to every event of this type"
    if name == "no_recent_mna_mandate":
        recent = [d for e in entities(g, gid) for d in g.ins(e, "EXECUTED_FOR")
                  if g.node(d.src)["props"]["product_code"] == "MNA"
                  and 0 <= (at - _d(g.node(d.src)["props"]["date"])).days < 540]
        return (not recent), (f"no M&A mandate with us in prior 18m" if not recent
                              else f"recent M&A mandate {recent[0].src}")
    if name == "leverage_headroom":
        lev = leverage(g, gid, at)
        return lev < 3.5, f"leverage {lev:.1f}x vs 3.5x threshold (credit paper / financials)"
    if name == "cross_currency":
        ok = bool(cc) and COUNTRIES[cc]["currency"] != p["functional_ccy"]
        return ok, f"{COUNTRIES[cc]['currency'] if cc else '?'} exposure vs functional ccy {p['functional_ccy']}"
    if name == "no_tb_presence_in_country":
        has = [h for h in holdings(g, gid, at) if h["country"] == cc and h["product_code"] == "CASH_MGMT"]
        return (not has), (f"no operating account with us in {COUNTRIES[cc]['name']}" if not has
                           else f"already bank {has[0]['entity']} in {COUNTRIES[cc]['name']}")
    if name == "investment_grade":
        return p["risk_rating"] in INVESTMENT_GRADE, f"internal rating {p['risk_rating']}"
    if name == "sub_investment_grade":
        return p["risk_rating"] not in INVESTMENT_GRADE, f"internal rating {p['risk_rating']}"
    if name == "no_rates_hedge":
        return "IRS_HEDGE" not in held, "no IRS hedge held" if "IRS_HEDGE" not in held else "IRS hedge already held"
    if name == "trade_intensive_industry":
        return p["industry"] in TRADE_INTENSIVE, f"industry {p['industry']}"
    if name == "commodity_intensive_industry":
        return p["industry"] in COMMODITY_INTENSIVE, f"industry {p['industry']}"
    if name == "no_scf_product":
        return "SCF" not in held, "no SCF programme" if "SCF" not in held else "SCF already in place"
    if name == "no_liquidity_product":
        return "LIQUIDITY" not in held, ("no liquidity structure" if "LIQUIDITY" not in held
                                         else "liquidity structure already in place")
    if name == "has_cash_mgmt_elsewhere":
        return "CASH_MGMT" in held, "group already uses our cash management"
    raise KeyError(name)


def event_record(g, eid) -> dict:
    p = g.node(eid)["props"]
    return {"event_id": eid, "group_id": g.outs(eid, "AFFECTS")[0].dst, "date": p["date"],
            "event_type": p["event_type"], "headline": p["headline"], "country": p.get("country", ""),
            "amount_usd": p.get("amount_usd", 0),
            "counterparty": [g.name(e.dst) for e in g.outs(eid, "INVOLVES")]}


def event_opportunities(g: KnowledgeGraph, eid: str) -> dict:
    """Evaluate the ontology playbook for an event against the graph *as of the event date*."""
    ev = event_record(g, eid)
    gid, at = ev["group_id"], _d(ev["date"])
    recs, rejected = [], []
    for rule in EVENT_PLAYBOOK[ev["event_type"]]:
        ok, why = _cond(g, rule["condition"], gid, ev, at)
        item = {"product_code": rule["product"], "product": pname(rule["product"]) if rule["product"] else None,
                "family": PRODUCT_BY_CODE[rule["product"]]["family"] if rule["product"] else "Relationship",
                "rule": rule["why"], "condition": rule["condition"], "evidence": why}
        (recs if ok else rejected).append(item)
    extra = []
    if ev["event_type"] == "MGMT_CHANGE":
        extra = new_executive_context(g, gid)
    return {"event": ev, "recommended": recs, "rejected_by_context": rejected, "relationship_context": extra}


def baseline_event_opportunities(g, eid) -> list[str]:
    """Today's news alert: event-type keyword -> generic product list (no client context)."""
    ev = event_record(g, eid)
    return [r["product"] for r in EVENT_PLAYBOOK[ev["event_type"]] if r["product"]]


def new_executive_context(g, gid) -> list[str]:
    out = []
    for c in contacts(g, gid):
        for pw in g.outs(c["contact_id"], "PREVIOUSLY_WORKED_AT"):
            org = pw.dst
            og = group_of(g, org)
            line = f"{c['name']} ({c['title']}) previously {pw.props.get('title', '')} at {g.name(org)}"
            if og:
                top = strongest_bankers_for_group(g, og)
                if top:
                    line += f" - a client where {top[0][0]} has {top[0][1]} touchpoints; ask for a warm intro"
            out.append(line)
    return out


# ---------------------------------------------------------------------------
# Relationship intelligence (Experiment 3)
# ---------------------------------------------------------------------------
def contacts(g, gid) -> list[dict]:
    out = []
    for e in entities(g, gid):
        for w in g.ins(e, "WORKS_AT"):
            p = g.node(w.src)["props"]
            out.append({"contact_id": w.src, "name": p["name"], "title": p["title"], "entity": g.name(e)})
    return out


def contact_touchpoints(g, cid) -> list[dict]:
    out = []
    for i in g.ins(cid, "WITH_CONTACT"):
        ip = g.node(i.src)["props"]
        b = g.outs(i.src, "BY_BANKER")[0].dst
        out.append({"date": ip["date"], "channel": ip["channel"], "banker_id": b, "banker": g.name(b)})
    return sorted(out, key=lambda x: x["date"], reverse=True)


def uncovered_contacts(g, gid, days: int = 180) -> list[dict]:
    out = []
    for c in contacts(g, gid):
        tps = contact_touchpoints(g, c["contact_id"])
        last = tps[0] if tps else None
        if not last or (TODAY - _d(last["date"])).days > days:
            out.append({**c, "last_touchpoint": last["date"] if last else None,
                        "last_banker": last["banker"] if last else None})
    return out


def strongest_bankers_for_contact(g, cid) -> list[tuple[str, int, str]]:
    cnt, last = Counter(), {}
    for t in contact_touchpoints(g, cid):
        cnt[t["banker"]] += 1
        last.setdefault(t["banker"], t["date"])
    return [(b, n, last[b]) for b, n in cnt.most_common()]


def strongest_bankers_for_group(g, gid) -> list[tuple[str, int]]:
    cnt = Counter()
    for i in g.ins(gid, "ABOUT"):
        if g.type(i.src) == "Interaction":
            cnt[g.name(g.outs(i.src, "BY_BANKER")[0].dst)] += 1
    return cnt.most_common()


def entities_with_product(g, gid, code) -> list[dict]:
    out = []
    for h in holdings(g, gid):
        if h["product_code"] == code:
            cov = [e.src for e in g.ins(gid, "COVERS") if e.props["role"] == "Lead RM"]
            out.append({"entity": h["entity"], "country": h["country"], "since": h["since"],
                        "coverage_banker": g.name(cov[0]) if cov else None})
    return out


PATH_KIND = [("JV_PARTNER_WITH", "JV"), ("DIRECTOR_OF", "SHARED_DIRECTOR"),
             ("PREVIOUSLY_WORKED_AT", "FORMER_EMPLOYEE"), ("PAYS", "SUPPLIER"), ("SUPPLIER_OF", "SUPPLIER"),
             ("TRADES_WITH", "TRADE"), ("DEAL_COUNTERPARTY", "DEAL"), ("INVOLVES", "EVENT"),
             ("MENTIONS", "DOC_MENTION"), ("WORKS_AT", "EMPLOYEE")]
CONNECT_AVOID = {"Country", "Industry", "Currency", "Product", "Intent", "WalletEstimate", "Interaction", "Banker",
                 "Opportunity"}


def _org_nodes(g, nid) -> list[str]:
    """A client group is represented by itself plus all its legal entities."""
    if g.type(nid) == "ClientGroup":
        return [nid] + entities(g, nid)
    return [nid]


def find_connections(g: KnowledgeGraph, a: str, b: str, max_hops: int = 3) -> dict:
    """Enumerate explainable paths between two organisations (multi-hop)."""
    starts, targets = _org_nodes(g, a), set(_org_nodes(g, b))
    found: dict[tuple, dict] = {}
    for s in starts:
        stack = [(s, [s])]
        while stack:
            cur, path = stack.pop()
            hops = (len(path) - 1) // 2
            if hops >= max_hops:
                continue
            for e, nb, d in g.neighbours(cur):
                if nb in path[::2] or e.rel in ("BELONGS_TO_GROUP", "ULTIMATE_PARENT", "SUBSIDIARY_OF",
                                                "SUPPLIER_OF", "TRADES_WITH"):
                    continue
                if nb in starts:
                    continue
                newp = path + [(e.rel, d, e.props), nb]
                if nb in targets:
                    outsiders = [n for n in newp[2:-1:2] if g.type(n) in ("LegalEntity", "ExternalCompany")
                                 and n not in targets and n not in starts]
                    kind, via = ("INDIRECT", g.name(outsiders[0])) if outsiders else _classify(g, newp)
                    key = (kind, via)
                    if key not in found or len(newp) < len(found[key]["path"]):
                        found[key] = {"kind": kind, "via": via, "path": newp, "explanation": explain_path(g, newp)}
                    continue
                if g.type(nb) in CONNECT_AVOID or (g.type(nb) == "ClientGroup" and nb not in targets):
                    continue
                stack.append((nb, newp))
    conns = sorted(found.values(), key=lambda x: (x["kind"] == "INDIRECT", len(x["path"]), x["kind"]))
    return {"a": g.name(a), "b": g.name(b), "connections": [
        {"kind": c["kind"], "via": c["via"], "hops": (len(c["path"]) - 1) // 2, "explanation": c["explanation"],
         "nodes": c["path"][::2]} for c in conns]}


def _classify(g, path) -> tuple[str, str]:
    rels = [step[0] for step in path[1::2]]
    for rel, kind in PATH_KIND:
        if rel in rels:
            i = rels.index(rel)
            step = path[1::2][i]
            if kind == "JV":
                return kind, step[2].get("jv_name", "")
            if kind in ("SHARED_DIRECTOR", "FORMER_EMPLOYEE"):
                return kind, g.name(path[2 * i] if g.type(path[2 * i]) == "Contact" else path[2 * i + 2])
            if kind == "SUPPLIER":
                payer = next((n for n in path[::2] if g.type(n) == "LegalEntity"), path[0])
                return kind, g.name(payer)
            if kind in ("DEAL", "EVENT", "DOC_MENTION"):
                mid = path[2 * i] if g.type(path[2 * i]) in ("Deal", "CorporateEvent", "CallReport", "CreditPaper") \
                    else path[2 * i + 2]
                return kind, mid
            return kind, ""
    return "OTHER", ""


def explain_path(g, path) -> str:
    parts = [g.name(path[0])]
    for i in range(1, len(path), 2):
        rel, d, props = path[i]
        arrow = f"-[{rel}]->" if d == "out" else f"<-[{rel}]-"
        extra = ""
        if rel == "PAYS":
            extra = f" USD {props.get('annual_usd', 0)/1e6:,.1f}m/yr {props.get('currency', '')} {props.get('purpose', '')}"
        if rel == "JV_PARTNER_WITH":
            extra = f" {props.get('jv_name', '')}"
        parts.append(f"{arrow}{('(' + extra.strip() + ')') if extra else ''} {g.name(path[i + 1])}")
    return " ".join(parts)


def related_parties(g: KnowledgeGraph, gid: str) -> list[dict]:
    out, seen = [], set()

    def add(nid, rel, why):
        if nid in seen or nid == gid:
            return
        seen.add(nid)
        out.append({"id": nid, "name": g.name(nid), "type": g.type(nid), "relation": rel, "why": why})
    ents = entities(g, gid)
    for e in ents:
        for j in g.neighbours(e, {"JV_PARTNER_WITH"}):
            add(j[1], "JV partner", f"{j[0].props.get('jv_name')} with {g.name(e)}")
    for s in g.ins(gid, "SUPPLIER_OF"):
        add(s.src, "supplier", "inferred from supplier payments")
    for t in g.neighbours(gid, {"TRADES_WITH"}):
        add(t[1], "trading partner", "client-to-client payment flows")
    for c in contacts(g, gid):
        for r in g.outs(c["contact_id"], "PREVIOUSLY_WORKED_AT") + g.outs(c["contact_id"], "DIRECTOR_OF"):
            add(r.dst, "people link", f"{c['name']} {r.rel.lower().replace('_', ' ')} {g.name(r.dst)}")
    ind = g.outs(gid, "IN_INDUSTRY")[0].dst
    for p in g.ins(ind, "IN_INDUSTRY"):
        if g.type(p.src) == "ClientGroup":
            add(p.src, "industry peer", f"same industry ({g.name(ind)})")
    return out


def deals_for_org(g, nid) -> list[dict]:
    targets = _org_nodes(g, nid) if g.type(nid) == "ClientGroup" else [nid]
    if g.type(nid) == "LegalEntity":
        targets = _org_nodes(g, group_of(g, nid))
    out = []
    for t in targets:
        for d in g.ins(t, "EXECUTED_FOR"):
            p = g.node(d.src)["props"]
            out.append({"deal_id": d.src, "date": p["date"], "product_code": p["product_code"],
                        "product": pname(p["product_code"]), "size_usd": p["size_usd"], "summary": p["summary"],
                        "client": g.name(t)})
        for d in g.ins(t, "DEAL_COUNTERPARTY"):
            p = g.node(d.src)["props"]
            out.append({"deal_id": d.src, "date": p["date"], "product_code": p["product_code"],
                        "product": pname(p["product_code"]), "size_usd": p["size_usd"], "summary": p["summary"],
                        "client": f"(counterparty) {g.name(t)}"})
    return out


# ---------------------------------------------------------------------------
# Portfolio change detection (agentic question)
# ---------------------------------------------------------------------------
def flow_trend(g, gid) -> dict | None:
    series = defaultdict(int)
    for _e, p in _flows(g, gid):
        for m, v in zip(p.props["months"], p.props["monthly_usd"]):
            series[m] += v
    months = sorted(series)
    if len(months) < 8:
        return None
    last2 = sum(series[m] for m in months[-2:]) / 2
    prior = sum(series[m] for m in months[-8:-2]) / 6
    chg = last2 / prior - 1 if prior else 0
    return {"months": months, "series": [series[m] for m in months], "last2_avg": last2, "prior6_avg": prior,
            "change": round(chg, 3)}


def portfolio_changes(g: KnowledgeGraph, banker_id: str, days: int = 60) -> dict:
    groups = [e.dst for e in g.outs(banker_id, "COVERS") if e.props["role"] == "Lead RM"] or \
             [e.dst for e in g.outs(banker_id, "COVERS")]
    since = TODAY - timedelta(days=days)
    items = []
    for gid in groups:
        name = g.name(gid)
        for e in g.ins(gid, "AFFECTS"):
            ev = event_record(g, e.src)
            if _d(ev["date"]) >= since:
                opp = event_opportunities(g, e.src)
                prods = [r["product"] for r in opp["recommended"] if r["product"]]
                prio = 3 if ev["event_type"] in ("ACQUISITION", "RATING_DOWNGRADE", "BOND_MATURITY") else 2
                action = ("Convene deal team; pitch " + ", ".join(prods)) if prods else "; ".join(
                    r["rule"] for r in opp["recommended"])
                if ev["event_type"] == "RATING_DOWNGRADE":
                    action = "Credit review with risk; hold new leverage asks; check covenant headroom"
                items.append({"group_id": gid, "client": name, "kind": "EVENT", "priority": prio, "date": ev["date"],
                              "what": ev["headline"], "action": action,
                              "evidence": [r["evidence"] for r in opp["recommended"] if r["condition"] != "always"]
                              + opp["relationship_context"],
                              "node_ids": [e.src, gid]})
        ft = flow_trend(g, gid)
        if ft and (ft["change"] <= -0.35 or ft["change"] >= 0.5):
            drop = ft["change"] < 0
            comp = [d.src for d in g.ins(gid, "ABOUT") if g.type(d.src) == "CallReport" and
                    any(x.dst == "INTENT:ATTRITION_RISK" for x in g.outs(d.src, "EXPRESSES"))]
            items.append({"group_id": gid, "client": name, "kind": "FLOW_DROP" if drop else "FLOW_SPIKE",
                          "priority": 3 if drop else 2, "date": ft["months"][-1],
                          "what": f"Payment flows {ft['change']:+.0%} (last 2m avg vs prior 6m)",
                          "action": ("Retention call with Treasurer; review pricing vs competitor" if drop else
                                     "Understand driver of spike; offer liquidity / FX on new volumes"),
                          "evidence": [f"Last 2m avg USD {ft['last2_avg']/1e6:,.1f}m vs prior USD "
                                       f"{ft['prior6_avg']/1e6:,.1f}m"]
                          + ([f"Competitor pricing mentioned in {', '.join(comp)}"] if comp else []),
                          "node_ids": [gid] + comp})
        seen_intents = set()
        for d in sorted(g.ins(gid, "ABOUT"), key=lambda d: g.node(d.src)["props"].get("date", ""), reverse=True):
            if g.type(d.src) != "CallReport":
                continue
            dd = _d(g.node(d.src)["props"]["date"])
            if dd < since:
                continue
            for x in g.outs(d.src, "EXPRESSES"):
                code = g.node(x.dst)["props"].get("product")
                if code and x.props["polarity"] == "positive" and code not in held_codes(g, gid) \
                        and x.dst not in seen_intents:
                    seen_intents.add(x.dst)
                    items.append({"group_id": gid, "client": name, "kind": "CLIENT_INTENT", "priority": 2,
                                  "date": dd.isoformat(), "what": f"Client intent: {g.name(x.dst)}",
                                  "action": f"Follow up with {pname(code)} specialist",
                                  "evidence": [f"{d.src}: \"{x.props['sentence']}\""], "node_ids": [d.src, gid]})
        for o in g.outs(gid, "HAS_OPPORTUNITY"):
            p = g.node(o.dst)["props"]
            if _d(p["updated"]) >= since:
                items.append({"group_id": gid, "client": name, "kind": "PIPELINE", "priority": 1,
                              "date": p["updated"], "what": f"{pname(p['product_code'])} moved/updated at stage "
                                                            f"{p['stage']}",
                              "action": "Keep momentum: confirm next step & owner", "evidence": [o.dst],
                              "node_ids": [o.dst, gid]})
        stale = [c for c in uncovered_contacts(g, gid, 180) if "CFO" in c["title"] or "Treasurer" in c["title"]]
        for c in stale:
            items.append({"group_id": gid, "client": name, "kind": "COVERAGE_GAP", "priority": 1,
                          "date": c["last_touchpoint"] or "", "what": f"No touchpoint with {c['name']} ({c['title']})"
                                                                     f" for >180 days",
                          "action": f"Schedule meeting (last met by {c['last_banker'] or 'nobody'})",
                          "evidence": [f"last touchpoint {c['last_touchpoint']}"],
                          "node_ids": [c["contact_id"], gid]})
    items.sort(key=lambda x: (-x["priority"], x["date"]), reverse=False)
    items.sort(key=lambda x: -x["priority"])
    return {"banker": g.name(banker_id), "banker_id": banker_id, "window_days": days,
            "clients": [g.name(x) for x in groups], "changes": items}
