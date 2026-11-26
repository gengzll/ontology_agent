"""Three lighthouse experiments from the exploration brief.

  1. Cross-sell / Next Best Action  - 25 clients, payments + holdings + wallet + pipeline + call reports,
                                      benchmarked against today's opportunity identification.
  2. Event -> Opportunity           - historical corporate events; can the graph reconstruct the GB/TB/Markets
                                      opportunities that actually followed? precision / recall (+ acceptance log).
  3. Relationship Intelligence      - 15 complex client groups; multi-hop questions answered end-to-end by the
                                      agent vs. a document-RAG baseline (today's Client Advisor).

Ground truth comes from data/ground_truth/ (written by the generator, never read by the agent).

Run:  python -m ontology_agent.experiments.run
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import date, timedelta

from .. import AS_OF, DATA_DIR, REPORT_DIR
from ..agent import offline
from ..engines import analytics as A
from ..engines.vector import index
from ..kg.build import load_graph, read_csv

GT = DATA_DIR / "ground_truth"
TODAY = date.fromisoformat(AS_OF)


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "tp": tp, "fp": fp, "fn": fn}


# ---------------------------------------------------------------------------
# Experiment 1
# ---------------------------------------------------------------------------
def exp1_nba(k: int = 3) -> dict:
    g, _ = load_graph()
    rows = list(csv.DictReader(open(GT / "nba_realised_next12m.csv")))
    cohort = sorted({r["group_id"] for r in rows} | {f"G{i:03d}" for i in range(1, 26)})
    truth = defaultdict(set)
    declined = defaultdict(set)
    for r in rows:
        if r["realised"] == "1":
            truth[r["group_id"]].add(r["product_code"])
        if r["driver"] == "DECLINED":
            declined[r["group_id"]].add(r["product_code"])
    pipeline = defaultdict(set)
    for o in read_csv("crm_pipeline.csv"):
        pipeline[o["group_id"]].add(o["product_code"])

    def evaluate(recommender) -> dict:
        tp = fp = fn = hit_clients = net_new = declined_recs = 0
        per = {}
        for gid in cohort:
            recs = recommender(gid)
            t = truth[gid]
            hits = set(recs) & t
            tp += len(hits)
            fp += len(set(recs) - t)
            fn += len(t - set(recs))
            hit_clients += bool(hits)
            net_new += len(hits - pipeline[gid])
            declined_recs += len(set(recs) & declined[gid])
            per[gid] = {"recommended": recs, "realised": sorted(t), "hits": sorted(hits)}
        m = prf(tp, fp, fn)
        with_truth = sum(1 for gid in cohort if truth[gid])
        pk = tp / (k * len(cohort))
        f1k = 2 * pk * m["recall"] / (pk + m["recall"]) if pk + m["recall"] else 0.0
        m.update({"precision_at_k": round(pk, 3), "f1_at_k": round(f1k, 3),
                  "client_hit_rate": round(hit_clients / max(1, with_truth), 3),
                  "net_new_hits_not_in_pipeline": net_new, "recommended_declined_products": declined_recs})
        return {"metrics": m, "per_client": per}

    graph = evaluate(lambda gid: [r["product_code"] for r in A.next_best_actions(g, gid, k)])
    base = evaluate(lambda gid: A.baseline_nba(g, gid, k))
    ablations = {}
    for name, srcs in {"payments_only": {"payments", "payments+ownership"},
                       "call_reports_only": {"call reports"},
                       "holdings+wallet+pipeline_only": {"holdings", "holdings+ownership", "wallet", "pipeline",
                                                         "peers"}}.items():
        ablations[name] = evaluate(lambda gid, s=srcs: [r["product_code"]
                                                        for r in A.next_best_actions(g, gid, k, s)])["metrics"]
    expl = [len(r["sources"]) for gid in cohort for r in A.next_best_actions(g, gid, k)]
    return {"cohort_size": len(cohort), "k": k, "realised_opportunities": sum(len(v) for v in truth.values()),
            "graph_nba": graph["metrics"], "baseline_today": base["metrics"], "ablations": ablations,
            "avg_sources_per_graph_recommendation": round(sum(expl) / max(1, len(expl)), 2),
            "examples": {gid: {"graph": graph["per_client"][gid], "baseline": base["per_client"][gid]["recommended"]}
                         for gid in cohort[:6]}}


# ---------------------------------------------------------------------------
# Experiment 2
# ---------------------------------------------------------------------------
def exp2_events() -> dict:
    g, _ = load_graph()
    truth = defaultdict(set)
    hist = set()
    for r in csv.DictReader(open(GT / "event_outcomes.csv")):
        hist.add(r["event_id"])
        if r["product_code"]:
            truth[r["event_id"]].add(r["product_code"])
    events = sorted(hist)
    tot = {"graph": [0, 0, 0], "baseline": [0, 0, 0]}
    by_type = defaultdict(lambda: {"graph": [0, 0, 0], "baseline": [0, 0, 0], "n": 0})
    explain = 0
    n_recs = 0
    examples = []
    for eid in events:
        res = A.event_opportunities(g, eid)
        et = res["event"]["event_type"]
        gp = {x["product_code"] for x in res["recommended"] if x["product_code"]}
        bp = set(A.baseline_event_opportunities(g, eid))
        t = truth[eid]
        by_type[et]["n"] += 1
        for name, pred in (("graph", gp), ("baseline", bp)):
            c = [len(pred & t), len(pred - t), len(t - pred)]
            for i in range(3):
                tot[name][i] += c[i]
                by_type[et][name][i] += c[i]
        n_recs += len(gp)
        explain += sum(1 for x in res["recommended"] if x["product_code"] and x["evidence"])
        if len(examples) < 6 and res["rejected_by_context"]:
            examples.append({"event": res["event"]["headline"], "event_id": eid, "graph": sorted(gp),
                             "baseline": sorted(bp), "actual": sorted(t),
                             "rejected": [f"{x['product_code']}: {x['evidence']}" for x in res["rejected_by_context"]]})
    acceptance = feedback_acceptance()
    return {"events_evaluated": len(events), "graph": prf(*tot["graph"]), "baseline_news_alert": prf(*tot["baseline"]),
            "by_event_type": {k: {"n": v["n"], "graph": prf(*v["graph"]), "baseline": prf(*v["baseline"])}
                              for k, v in sorted(by_type.items())},
            "recommendations_with_context_evidence": f"{explain}/{n_recs}",
            "banker_acceptance": acceptance, "examples": examples}


def feedback_acceptance() -> dict:
    path = DATA_DIR / "feedback.jsonl"
    if not path.exists():
        return {"status": "no feedback captured yet - use the thumbs up/down buttons in the web UI during the pilot"}
    rows = [json.loads(x) for x in open(path) if x.strip()]
    by_route = defaultdict(Counter)
    for r in rows:
        by_route[r.get("route", "?")][r.get("verdict", "?")] += 1
    return {"total": len(rows), "by_route": {k: {**v, "acceptance": round(v["accept"] / max(1, sum(v.values())), 3)}
                                             for k, v in by_route.items()}}


# ---------------------------------------------------------------------------
# Experiment 3
# ---------------------------------------------------------------------------
def _gold_relationship() -> list[dict]:
    """Gold answers computed straight from the source systems (+ planted ground truth)."""
    groups = {r["group_id"]: r for r in read_csv("crm_client_groups.csv")}
    ents = read_csv("crm_legal_entities.csv")
    ent_group = {e["entity_id"]: e["group_id"] for e in ents}
    names = {e["entity_id"]: e["name"] for e in ents}
    holdings = read_csv("product_holdings.csv")
    contacts = read_csv("crm_contacts.csv")
    inter = read_csv("crm_interactions.csv")
    bankers = {b["banker_id"]: b["name"] for b in read_csv("crm_bankers.csv")}
    deals = read_csv("deals_history.csv")
    jvs = read_csv("ref_joint_ventures.csv")
    pays = read_csv("payments_monthly_flows.csv")
    alias = {r["raw_name"]: r["entity_id"] for r in csv.DictReader(open(GT / "payment_alias_truth.csv"))}
    ext_names = {x["name"] for x in read_csv("ref_external_companies.csv")}
    plants = json.load(open(GT / "relationship_plants.json"))
    qs = []
    for pl in plants:
        gid = pl["group_id"]
        short = groups[gid]["name"].removesuffix(" Group")
        gents = {e["entity_id"] for e in ents if e["group_id"] == gid}
        # T1 entities holding cash management
        qs.append({"group_id": gid, "template": "T1_entities_with_product",
                   "question": f"Which legal entities of {short} hold cash management with us?",
                   "gold": sorted({names[h["entity_id"]] for h in holdings
                                   if h["entity_id"] in gents and h["product_code"] == "CASH_MGMT"})})
        # T2 connection
        qs.append({"group_id": gid, "template": "T2_connection",
                   "question": f"How are {short} and {pl['company_b']} connected?",
                   "gold": sorted((c["type"], c["via"]) for c in pl["connections"])})
        # T3 contacts not met in 180 days
        gc = [c for c in contacts if c["employer_id"] in gents]
        last = {}
        for i in inter:
            last[i["contact_id"]] = max(last.get(i["contact_id"], ""), i["date"])
        cutoff = (TODAY - timedelta(days=180)).isoformat()
        qs.append({"group_id": gid, "template": "T3_contacts_not_met",
                   "question": f"Which contacts at {short} have we not met in the last 6 months?",
                   "gold": sorted(c["name"] for c in gc if last.get(c["contact_id"], "") < cutoff)})
        # T4 strongest banker with CFO
        cfo = next(c for c in gc if c["title"] == "Group CFO")
        cnt = Counter(bankers[i["banker_id"]] for i in inter if i["contact_id"] == cfo["contact_id"])
        top = max(cnt.values()) if cnt else 0
        qs.append({"group_id": gid, "template": "T4_strongest_banker_cfo",
                   "question": f"Who has the strongest relationship with the CFO of {short}?",
                   "gold": sorted(b for b, n in cnt.items() if n == top)})
        # T5 deals for JV partners and suppliers
        partners_ext, partner_groups = set(), set()
        for j in jvs:
            if j["party_a"] in {names[e] for e in gents}:
                partners_ext.add(j["party_b"])
        for p in pays:
            if p["payer_entity_id"] in gents and p["purpose"] == "supplier":
                if p["beneficiary_name"] in alias:
                    og = ent_group[alias[p["beneficiary_name"]]]
                    if og != gid:
                        partner_groups.add(og)
                elif p["beneficiary_name"] in ext_names:
                    partners_ext.add(p["beneficiary_name"])
        gold = {d["deal_id"] for d in deals if ent_group[d["entity_id"]] in partner_groups
                or d["counterparty_name"] in partners_ext}
        qs.append({"group_id": gid, "template": "T5_deals_for_jv_and_suppliers",
                   "question": f"What deals have we executed for {short}'s JV partners and suppliers?",
                   "gold": sorted(gold)})
    return qs


def _baseline_rag_answer(q: dict, g) -> list:
    """Today's Client Advisor: top-5 retrieved documents, then pull candidate answers of the right type."""
    hits = index().search(q["question"], k=5)
    text = " ".join(h["text"] for h in hits)
    t = q["template"]
    if t == "T1_entities_with_product":
        return [g.name(e) for e in g.of_type("LegalEntity") if g.name(e) in text and "cash" in text.lower()]
    if t == "T2_connection":
        b = q["question"].split(" and ")[-1].removesuffix(" connected?")
        out = []
        for h in hits:
            if b in h["text"]:
                out.append(("DOC_MENTION", h["doc_id"]))
        return out
    if t == "T3_contacts_not_met":
        return []  # absence of interactions is not retrievable from text
    if t == "T4_strongest_banker_cfo":
        c = Counter(g.name(b) for b in g.of_type("Banker") if g.name(b) in text for _ in [0]
                    for _m in re.finditer(re.escape(g.name(b)), text))
        return [c.most_common(1)[0][0]] if c else []
    if t == "T5_deals_for_jv_and_suppliers":
        return [h["doc_id"] for h in hits if h["doc_type"] == "DealTombstone"]
    return []


def exp3_relationship() -> dict:
    g, _ = load_graph()
    qs = _gold_relationship()
    agg = {"graph": defaultdict(lambda: [0, 0, 0, 0, 0]), "baseline": defaultdict(lambda: [0, 0, 0, 0, 0])}
    examples = []
    for q in qs:
        res = offline.answer(q["question"])
        graph_items = res["answer_items"]
        if q["template"] == "T2_connection":
            graph_items = [tuple(x) for x in graph_items]
            # unplanted-but-real direct links (deals, docs, events) are not penalised
            graph_items = [x for x in graph_items if x[0] in {"SUPPLIER", "SHARED_DIRECTOR", "FORMER_EMPLOYEE", "JV"}]
        base_items = _baseline_rag_answer(q, g)
        gold = set(map(tuple, q["gold"])) if q["template"] == "T2_connection" else set(q["gold"])
        for name, items in (("graph", graph_items), ("baseline", base_items)):
            s = set(items)
            tp, fp, fn = len(s & gold), len(s - gold), len(gold - s)
            a = agg[name][q["template"]]
            a[0] += tp
            a[1] += fp
            a[2] += fn
            a[3] += int(s == gold)
            a[4] += 1
        if len(examples) < 5 and q["template"] in ("T2_connection", "T5_deals_for_jv_and_suppliers"):
            examples.append({"question": q["question"], "route": res["route"], "gold": q["gold"],
                             "graph": graph_items[:8], "baseline": base_items[:8]})

    def summarise(d):
        out = {}
        tot = [0, 0, 0, 0, 0]
        for t, a in sorted(d.items()):
            out[t] = {**prf(a[0], a[1], a[2]), "exact_match": f"{a[3]}/{a[4]}"}
            tot = [x + y for x, y in zip(tot, a)]
        out["ALL"] = {**prf(tot[0], tot[1], tot[2]), "exact_match": f"{tot[3]}/{tot[4]}"}
        return out
    return {"client_groups": len({q["group_id"] for q in qs}), "questions": len(qs),
            "graph_agent": summarise(agg["graph"]), "baseline_doc_rag": summarise(agg["baseline"]),
            "examples": examples}


# ---------------------------------------------------------------------------
def kg_quality() -> dict:
    g, rep = load_graph()
    truth = list(csv.DictReader(open(GT / "payment_alias_truth.csv")))
    from ..kg.build import EntityResolver
    er = EntityResolver()
    for r in read_csv("crm_legal_entities.csv"):
        er.add(r["entity_id"], r["name"], r["country"])
    for r in read_csv("ref_external_companies.csv"):
        er.add(r["ext_id"], r["name"], r["country"])
    cc = {r["beneficiary_name"]: r["beneficiary_country"] for r in read_csv("payments_monthly_flows.csv")}
    ok = wrong = miss = 0
    for t in truth:
        hit = er.resolve(t["raw_name"], cc.get(t["raw_name"], ""))
        if hit == t["entity_id"]:
            ok += 1
        elif hit is None:
            miss += 1
        else:
            wrong += 1
    return {"graph": rep["graph"], "inferred_edges": rep["inferred"], "extracted_intents": rep["extracted_intents"],
            "payment_beneficiary_resolution": {"aliases": len(truth), "correct": ok, "wrong": wrong, "unresolved": miss,
                                               "precision": round(ok / max(1, ok + wrong), 3),
                                               "recall": round(ok / max(1, len(truth)), 3)}}


def to_markdown(res: dict) -> str:
    e1, e2, e3, kq = res["exp1"], res["exp2"], res["exp3"], res["kg_quality"]
    L = ["# Lighthouse experiment results (synthetic data)", "",
         "> All numbers below come from **synthetic data** whose hidden ground truth was generated with the same "
         "kinds of signals the graph looks for. They show that the pipeline works end-to-end and how the comparison "
         "will be run on real data - they are **not** evidence for the hypothesis itself.", "",
         "## 0. Knowledge-graph build quality", "",
         f"- Nodes {kq['graph']['nodes']:,}, edges {kq['graph']['edges']:,}, ontology violations "
         f"{kq['graph']['ontology_violations']}", f"- Inferred edges: {kq['inferred_edges']}",
         f"- Intents extracted from call reports: {kq['extracted_intents']}",
         f"- Payment beneficiary entity resolution: precision {kq['payment_beneficiary_resolution']['precision']}, "
         f"recall {kq['payment_beneficiary_resolution']['recall']} "
         f"({kq['payment_beneficiary_resolution']['aliases']} free-text aliases)", "",
         f"## 1. Cross-sell / Next Best Action ({e1['cohort_size']} clients, top-{e1['k']})", "",
         "| Method | Precision@k | Recall | F1@k | Hit-rate (clients w/ outcome) | Net-new hits (not in pipeline) | Recommended a declined product |",
         "|---|---|---|---|---|---|---|"]
    for name, m in (("Graph NBA (all sources connected)", e1["graph_nba"]),
                    ("Today: pipeline + product-gap", e1["baseline_today"])):
        L.append(f"| {name} | {m['precision_at_k']} | {m['recall']} | {m['f1_at_k']} | {m['client_hit_rate']} | "
                 f"{m['net_new_hits_not_in_pipeline']} | {m['recommended_declined_products']} |")
    for name, m in e1["ablations"].items():
        L.append(f"| Ablation: {name} | {m['precision_at_k']} | {m['recall']} | {m['f1_at_k']} | "
                 f"{m['client_hit_rate']} | "
                 f"{m['net_new_hits_not_in_pipeline']} | {m['recommended_declined_products']} |")
    L += ["", f"Average distinct data sources behind each graph recommendation: "
              f"{e1['avg_sources_per_graph_recommendation']}", "",
          f"## 2. Event -> Opportunity ({e2['events_evaluated']} historical events)", "",
          "| Method | Precision | Recall | F1 |", "|---|---|---|---|",
          f"| Graph (playbook x client context as of event date) | {e2['graph']['precision']} | {e2['graph']['recall']} "
          f"| {e2['graph']['f1']} |",
          f"| Today: news alert -> generic product list | {e2['baseline_news_alert']['precision']} | "
          f"{e2['baseline_news_alert']['recall']} | {e2['baseline_news_alert']['f1']} |", "",
          "| Event type | n | Graph P / R | Baseline P / R |", "|---|---|---|---|"]
    for t, v in e2["by_event_type"].items():
        if v["graph"]["tp"] + v["graph"]["fp"] + v["graph"]["fn"] == 0:
            L.append(f"| {t} | {v['n']} | n/a (relationship / risk action, no product) | n/a |")
            continue
        L.append(f"| {t} | {v['n']} | {v['graph']['precision']} / {v['graph']['recall']} | "
                 f"{v['baseline']['precision']} / {v['baseline']['recall']} |")
    L += ["", f"Recommendations carrying a context reason: {e2['recommendations_with_context_evidence']}",
          f"Banker acceptance: {json.dumps(e2['banker_acceptance'])}", "", "Examples of context filtering:"]
    for x in e2["examples"][:4]:
        L.append(f"- `{x['event_id']}` {x['event']} - graph {x['graph']}, baseline {x['baseline']}, actual "
                 f"{x['actual']}; rejected: {'; '.join(x['rejected'])}")
    L += ["", f"## 3. Relationship Intelligence ({e3['client_groups']} complex groups, {e3['questions']} multi-hop "
              f"questions)", "", "| Question template | Graph agent P / R / exact | Doc-RAG baseline P / R / exact |",
          "|---|---|---|"]
    for t in e3["graph_agent"]:
        a, b = e3["graph_agent"][t], e3["baseline_doc_rag"][t]
        L.append(f"| {t} | {a['precision']} / {a['recall']} / {a['exact_match']} | "
                 f"{b['precision']} / {b['recall']} / {b['exact_match']} |")
    return "\n".join(L) + "\n"


def main():
    REPORT_DIR.mkdir(exist_ok=True)
    res = {"kg_quality": kg_quality(), "exp1": exp1_nba(), "exp2": exp2_events(), "exp3": exp3_relationship()}
    (REPORT_DIR / "experiment_results.json").write_text(json.dumps(res, indent=2, default=str))
    md = to_markdown(res)
    (REPORT_DIR / "experiment_results.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
