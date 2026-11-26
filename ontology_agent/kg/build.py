"""ETL: source systems -> ontology-conformant knowledge graph.

Steps
  1. Load reference + CRM masters (groups, entities, externals, people, products).
  2. Entity resolution of free-text names (payment beneficiaries, prior employers,
     board seats, JV parties) onto master organisations.
  3. Map transactional sources to relations (holdings, payments, deals, pipeline,
     wallet, events, interactions).
  4. Information extraction from documents (call reports): intents + mentions.
  5. Reasoner: materialise inferred relations defined in the ontology.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from functools import lru_cache

from .. import SOURCE_DIR
from ..ontology import COUNTRIES, PRODUCTS
from .store import KnowledgeGraph

LEGAL_TOKENS = {"PTE", "LTD", "LIMITED", "CO", "SDN", "BHD", "PVT", "PRIVATE", "INC", "GMBH", "KK", "PTY", "SA",
                "DE", "CV", "FZE", "PT", "GROUP", "THE"}
ALIAS_COUNTRY = {"(S)": "SINGAPORE", "(M)": "MALAYSIA"}
# Geography words are stripped from the match key; geography is matched separately via the country field.
GEO_TOKENS = ({w for c in COUNTRIES.values() for w in c["name"].upper().split()} | set(COUNTRIES)
              | {"SHANGHAI", "MEXICO", "UK", "S", "M"})


def read_csv(name: str) -> list[dict]:
    with open(SOURCE_DIR / name, newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(name: str) -> list[dict]:
    with open(SOURCE_DIR / "documents" / name) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Entity resolution
# ---------------------------------------------------------------------------
def normalise(name: str) -> str:
    s = name.upper()
    for k, v in ALIAS_COUNTRY.items():
        s = s.replace(k, f" {v} ")
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    toks = [t for t in s.split() if t not in LEGAL_TOKENS and t not in GEO_TOKENS]
    return " ".join(toks)


class EntityResolver:
    """Blocking on country + three match tiers: exact key -> token-set similarity -> abbreviation.

    Keys drop legal-form and geography tokens ("Pte Ltd", "(Singapore)"), so the
    country of the record is used to disambiguate siblings such as
    "Nimbusline Cloud Inc." (US) vs "Nimbusline Cloud GmbH" (DE).
    """

    def __init__(self):
        self.key_to_ids: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
        self.records: list[tuple[str, str, set[str], str, int]] = []  # id, raw, key tokens, country, priority
        self.exact_names: dict[str, str] = {}
        self.stats = defaultdict(int)
        self.unresolved: dict[str, int] = defaultdict(int)

    def add(self, oid: str, name: str, country: str = "", priority: int = 0):
        key = normalise(name)
        self.exact_names.setdefault(name.strip().upper(), oid)
        self.key_to_ids[key].append((oid, country, priority))
        raw = re.sub(r"[^A-Z0-9 ]", " ", name.upper())
        self.records.append((oid, raw, set(key.split()), country, priority))

    @staticmethod
    def _pick(cands: list[tuple[str, str, int]], country: str) -> str | None:
        if country:
            same = [c for c in cands if c[1] == country]
            if same:
                cands = same
            elif len(cands) > 1:
                return None
        cands = sorted(cands, key=lambda c: c[2])
        if len(cands) > 1 and cands[0][2] == cands[1][2]:
            return None  # ambiguous
        return cands[0][0] if cands else None

    def resolve(self, name: str, country: str = "") -> str | None:
        if name.strip().upper() in self.exact_names:
            self.stats["exact"] += 1
            return self.exact_names[name.strip().upper()]
        key = normalise(name)
        if key in self.key_to_ids:
            hit = self._pick(self.key_to_ids[key], country)
            if hit:
                self.stats["exact"] += 1
                return hit
        toks = set(key.split())
        scored = []
        for oid, _raw, nt, cc, pr in self.records:
            if nt and toks:
                s = len(toks & nt) / len(toks | nt)
                if s >= 0.75:
                    scored.append((oid, cc, pr + (1 - s)))
        if scored:
            hit = self._pick(scored, country)
            if hit:
                self.stats["fuzzy"] += 1
                return hit
        # abbreviation tier e.g. "AUR LOG SG" -> Aurora Logistics (Singapore) Pte Ltd
        parts = re.sub(r"[^A-Z0-9 ]", " ", name.upper()).split()
        if len(parts) >= 2 and parts[-1] in COUNTRIES:
            cc, abbr = parts[-1], parts[:-1]
            cands = []
            for oid, raw, _nt, rcc, pr in self.records:
                rt = raw.split()
                if rcc == cc and all(any(t.startswith(a) for t in rt) for a in abbr):
                    cands.append((oid, rcc, pr))
            hit = self._pick(cands, cc) if cands else None
            if hit:
                self.stats["abbreviation"] += 1
                return hit
        self.stats["unresolved"] += 1
        self.unresolved[name] += 1
        return None


# ---------------------------------------------------------------------------
# Information extraction from call reports (deterministic, auditable).
# In production this is where an LLM extractor would populate the same relations.
# ---------------------------------------------------------------------------
INTENT_PATTERNS = [
    (r"refinanc|debt capital markets|bond issue|issue bonds", "REFINANCE", "DCM"),
    (r"acquisition target|bolt-on acquisition|m&a ideas", "ACQUISITION_INTEREST", "MNA"),
    (r"\blisting\b|\bipo\b", "IPO_INTEREST", "ECM"),
    (r"letter of credit|trade finance|trade volumes", "TRADE_GROWTH", "TRADE_FIN"),
    (r"commodity exposure|input price volatility", "COMMODITY_RISK", "COMMODITY_HEDGE"),
    (r"fix the interest rate|floating-rate debt", "RATES_RISK", "IRS_HEDGE"),
    (r"fx hedging|hedging programmes|fx volatility", "FX_RISK", "FX_HEDGE"),
    (r"supplier financing|extend payment terms", "SUPPLIER_FINANCING", "SCF"),
    (r"cash pooling|cash is trapped", "TRAPPED_CASH", "LIQUIDITY"),
    (r"consolidate operating accounts|several local banks", "BANK_CONSOLIDATION", "CASH_MGMT"),
    (r"competitor offered sharper pricing", "ATTRITION_RISK", None),
]
INTENT_LABELS = {"REFINANCE": "Refinancing", "ACQUISITION_INTEREST": "Acquisition interest",
                 "IPO_INTEREST": "IPO interest", "TRADE_GROWTH": "Trade growth", "COMMODITY_RISK": "Commodity risk",
                 "RATES_RISK": "Interest-rate risk", "FX_RISK": "FX risk", "SUPPLIER_FINANCING": "Supplier financing",
                 "TRAPPED_CASH": "Trapped cash", "BANK_CONSOLIDATION": "Bank consolidation",
                 "ATTRITION_RISK": "Attrition risk (competitor pricing)"}
NEGATION = re.compile(r"not interested|no plan|declined|by policy", re.I)


def extract_intents(text: str) -> list[dict]:
    out = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if sent.startswith("Discussed recent news"):
            continue  # echo of a news headline, not a client statement
        for pat, kind, product in INTENT_PATTERNS:
            if re.search(pat, sent, re.I):
                out.append({"kind": kind, "product": product,
                            "polarity": "negative" if NEGATION.search(sent) else "positive", "sentence": sent})
    return out


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_graph() -> tuple[KnowledgeGraph, dict]:
    g = KnowledgeGraph()
    er = EntityResolver()
    report: dict = {}

    # --- reference data
    for cc, c in COUNTRIES.items():
        g.add_node(f"CCY:{c['currency']}", "Currency", code=c["currency"], name=c["currency"])
        g.add_node(f"CTY:{cc}", "Country", name=c["name"], code=cc, currency=c["currency"])
        g.add_edge(f"CTY:{cc}", "HAS_CURRENCY", f"CCY:{c['currency']}")
    for p in PRODUCTS:
        g.add_node(f"P:{p['code']}", "Product", **p)

    def industry(name):
        return g.add_node(f"IND:{name}", "Industry", name=name)

    # --- masters
    groups = read_csv("crm_client_groups.csv")
    for r in groups:
        g.add_node(r["group_id"], "ClientGroup", **{k: v for k, v in r.items() if k != "group_id"})
        g.add_edge(r["group_id"], "IN_INDUSTRY", industry(r["industry"]))
        er.add(r["group_id"], r["name"], r["hq_country"], priority=1)
    for r in read_csv("crm_legal_entities.csv"):
        g.add_node(r["entity_id"], "LegalEntity", name=r["name"], country=r["country"],
                   entity_type=r["entity_type"], lei=r["lei"], group_id=r["group_id"])
        g.add_edge(r["entity_id"], "BELONGS_TO_GROUP", r["group_id"])
        g.add_edge(r["entity_id"], "INCORPORATED_IN", f"CTY:{r['country']}")
        er.add(r["entity_id"], r["name"], r["country"])
    for r in read_csv("crm_legal_entities.csv"):
        if r["parent_entity_id"]:
            g.add_edge(r["entity_id"], "SUBSIDIARY_OF", r["parent_entity_id"], ownership_pct=float(r["ownership_pct"]))
    for r in read_csv("ref_external_companies.csv"):
        g.add_node(r["ext_id"], "ExternalCompany", name=r["name"], country=r["country"], industry=r["industry"],
                   kind=r["kind"])
        g.add_edge(r["ext_id"], "INCORPORATED_IN", f"CTY:{r['country']}")
        g.add_edge(r["ext_id"], "IN_INDUSTRY", industry(r["industry"]))
        er.add(r["ext_id"], r["name"], r["country"])
    for r in read_csv("crm_bankers.csv"):
        g.add_node(r["banker_id"], "Banker", name=r["name"], role=r["role"], business=r["business"])
    for r in read_csv("crm_coverage.csv"):
        g.add_edge(r["banker_id"], "COVERS", r["group_id"], role=r["role"])

    for r in read_csv("crm_contacts.csv"):
        g.add_node(r["contact_id"], "Contact", name=r["name"], title=r["title"])
        g.add_edge(r["contact_id"], "WORKS_AT", r["employer_id"], title=r["title"])
        if r["prior_employer_name"]:
            oid = er.resolve(r["prior_employer_name"])
            if oid and g.type(oid) == "ClientGroup":
                oid = _parent_entity(g, oid)
            if oid:
                g.add_edge(r["contact_id"], "PREVIOUSLY_WORKED_AT", oid, title=r["prior_title"],
                           until=r["prior_until"])
        for seat in filter(None, r["board_seats"].split(";")):
            oid = er.resolve(seat)
            if oid:
                g.add_edge(r["contact_id"], "DIRECTOR_OF", oid)
    for r in read_csv("ref_joint_ventures.csv"):
        a, b = er.resolve(r["party_a"]), er.resolve(r["party_b"])
        if a and b:
            g.add_edge(a, "JV_PARTNER_WITH", b, jv_name=r["jv_name"])

    # --- holdings
    for r in read_csv("product_holdings.csv"):
        g.add_edge(r["entity_id"], "HOLDS", f"P:{r['product_code']}", since=r["since"],
                   limit_usd=float(r["limit_usd"]), balance_usd=float(r["balance_usd"]))

    # --- payments (aggregate monthly rows into one PAYS edge per corridor)
    agg: dict[tuple, dict] = {}
    pay_er = {"rows": 0, "resolved": 0}
    payee_cache: dict[tuple, str | None] = {}
    unresolved_nodes = 0
    for r in read_csv("payments_monthly_flows.csv"):
        pay_er["rows"] += 1
        ck = (r["beneficiary_name"], r["beneficiary_country"])
        if ck not in payee_cache:
            payee_cache[ck] = er.resolve(*ck)
        payee = payee_cache[ck]
        if payee is None:
            payee = f"UNRES:{normalise(r['beneficiary_name'])}"
            if payee not in g.nodes:
                unresolved_nodes += 1
                g.add_node(payee, "ExternalCompany", name=r["beneficiary_name"], country=r["beneficiary_country"],
                           kind="unresolved payee")
        else:
            pay_er["resolved"] += 1
        if g.type(payee) == "ClientGroup":
            payee = _parent_entity(g, payee)
        k = (r["payer_entity_id"], payee, r["currency"], r["purpose"], r["beneficiary_bank"])
        a = agg.setdefault(k, {"annual_usd": 0, "monthly": defaultdict(int), "raw_names": set()})
        a["annual_usd"] += int(r["amount_usd"])
        a["monthly"][r["month"]] += int(r["amount_usd"])
        a["raw_names"].add(r["beneficiary_name"])
    for (payer, payee, ccy, purpose, bank), a in agg.items():
        months = sorted(a["monthly"])
        g.add_edge(payer, "PAYS", payee, currency=ccy, purpose=purpose, rail=bank, annual_usd=a["annual_usd"],
                   monthly_usd=[a["monthly"][m] for m in months], months=months, raw_names=sorted(a["raw_names"]))
    report["entity_resolution"] = {**dict(er.stats), "payment_rows": pay_er["rows"],
                                   "payment_rows_resolved": pay_er["resolved"],
                                   "unresolved_payee_nodes": unresolved_nodes,
                                   "unresolved_examples": sorted(er.unresolved)[:10]}

    # --- wallet, financials (as node props), pipeline, deals, events
    fin = defaultdict(list)
    for r in read_csv("finance_financials.csv"):
        fin[r["group_id"]].append({k: float(v) for k, v in r.items() if k != "group_id"})
    for gid, rows in fin.items():
        g.nodes[gid]["props"]["financials"] = sorted(rows, key=lambda x: x["year"])
    for r in read_csv("wallet_estimates.csv"):
        wid = f"W:{r['group_id']}:{r['family']}:{r['year']}"
        w, rev = float(r["wallet_usd"]), float(r["our_revenue_usd"])
        g.add_node(wid, "WalletEstimate", family=r["family"], year=int(r["year"]), wallet_usd=w,
                   our_revenue_usd=rev, share=round(rev / w, 3) if w else 0,
                   name=f"{r['family']} wallet {r['year']}")
        g.add_edge(r["group_id"], "HAS_WALLET", wid)
    for r in read_csv("crm_pipeline.csv"):
        g.add_node(r["opp_id"], "Opportunity", stage=r["stage"], est_revenue_usd=float(r["est_revenue_usd"]),
                   created=r["created"], updated=r["updated"], status=r["status"], product_code=r["product_code"],
                   name=f"{r['product_code']} opportunity ({r['stage']})")
        g.add_edge(r["group_id"], "HAS_OPPORTUNITY", r["opp_id"])
        g.add_edge(r["opp_id"], "FOR_PRODUCT", f"P:{r['product_code']}")
        g.add_edge(r["opp_id"], "OWNED_BY", r["owner_banker_id"])
    for r in read_csv("news_corporate_events.csv"):
        g.add_node(r["event_id"], "CorporateEvent", event_type=r["event_type"], date=r["date"],
                   headline=r["headline"], amount_usd=float(r["amount_usd"]), country=r["country"],
                   name=r["headline"])
        g.add_edge(r["event_id"], "AFFECTS", r["group_id"])
        if r["country"]:
            g.add_edge(r["event_id"], "IN_COUNTRY", f"CTY:{r['country']}")
        if r["counterparty_name"]:
            oid = er.resolve(r["counterparty_name"], r["country"])
            if oid:
                g.add_edge(r["event_id"], "INVOLVES", oid)
    for r in read_csv("deals_history.csv"):
        g.add_node(r["deal_id"], "Deal", date=r["date"], size_usd=float(r["size_usd"]),
                   revenue_usd=float(r["revenue_usd"]), summary=r["summary"], product_code=r["product_code"],
                   name=f"{r['deal_id']} {r['product_code']} {r['date']}")
        g.add_edge(r["deal_id"], "EXECUTED_FOR", r["entity_id"])
        g.add_edge(r["deal_id"], "FOR_PRODUCT", f"P:{r['product_code']}")
        for b in r["team"].split(";"):
            g.add_edge(r["deal_id"], "DEAL_TEAM", b)
        if r["event_id"]:
            g.add_edge(r["deal_id"], "ORIGINATED_FROM", r["event_id"])
        if r["counterparty_name"]:
            oid = er.resolve(r["counterparty_name"])
            if oid:
                g.add_edge(r["deal_id"], "DEAL_COUNTERPARTY", oid)

    # --- interactions & documents
    for r in read_csv("crm_interactions.csv"):
        g.add_node(r["interaction_id"], "Interaction", channel=r["channel"], date=r["date"],
                   name=f"{r['channel']} {r['date']}")
        g.add_edge(r["interaction_id"], "WITH_CONTACT", r["contact_id"])
        g.add_edge(r["interaction_id"], "BY_BANKER", r["banker_id"])
        g.add_edge(r["interaction_id"], "ABOUT", r["group_id"])
    org_names = {nid: g.name(nid) for t in ("ExternalCompany", "ClientGroup") for nid in g.of_type(t)
                 if not nid.startswith("UNRES:")}
    n_intents = 0
    for doc in read_jsonl("call_reports.jsonl") + read_jsonl("credit_papers.jsonl"):
        g.add_node(doc["doc_id"], doc["doc_type"], title=doc["title"], date=doc["date"], text=doc["text"],
                   group_id=doc["group_id"])
        g.add_edge(doc["doc_id"], "ABOUT", doc["group_id"])
        g.add_edge(doc["doc_id"], "AUTHORED_BY", doc["banker_id"])
        for c in doc["attendee_contact_ids"]:
            g.add_edge(doc["doc_id"], "ATTENDEE", c)
        for oid, nm in org_names.items():
            if oid != doc["group_id"] and nm in doc["text"]:
                g.add_edge(doc["doc_id"], "MENTIONS", oid)
        for cc, c in COUNTRIES.items():
            if c["name"] in doc["text"]:
                g.add_edge(doc["doc_id"], "MENTIONS", f"CTY:{cc}")
        if doc["doc_type"] == "CallReport":
            for it in extract_intents(doc["text"]):
                iid = f"INTENT:{it['kind']}"
                g.add_node(iid, "Intent", kind=it["kind"], name=INTENT_LABELS.get(it["kind"], it["kind"]),
                           product=it["product"])
                g.add_edge(doc["doc_id"], "EXPRESSES", iid, polarity=it["polarity"], sentence=it["sentence"],
                           date=doc["date"])
                if it["product"]:
                    g.add_edge(doc["doc_id"], "MENTIONS", f"P:{it['product']}")
                n_intents += 1
    report["extracted_intents"] = n_intents

    # --- reasoner
    report["inferred"] = materialise_inferences(g)
    report["graph"] = g.stats()
    return g, report


def _parent_entity(g: KnowledgeGraph, gid: str) -> str:
    ents = [e.src for e in g.ins(gid, "BELONGS_TO_GROUP")]
    for e in ents:
        if not g.outs(e, "SUBSIDIARY_OF"):
            return e
    return ents[0]


def group_of(g: KnowledgeGraph, nid: str) -> str | None:
    if g.type(nid) == "ClientGroup":
        return nid
    if g.type(nid) == "LegalEntity":
        return g.outs(nid, "BELONGS_TO_GROUP")[0].dst
    return None


def materialise_inferences(g: KnowledgeGraph) -> dict:
    counts = defaultdict(int)
    # ULTIMATE_PARENT (transitive closure of SUBSIDIARY_OF)
    for e in g.of_type("LegalEntity"):
        cur, seen = e, set()
        while True:
            up = g.outs(cur, "SUBSIDIARY_OF")
            if not up or cur in seen:
                break
            seen.add(cur)
            cur = up[0].dst
        if cur != e:
            g.add_edge(e, "ULTIMATE_PARENT", cur)
            counts["ULTIMATE_PARENT"] += 1
    # SUPPLIER_OF and TRADES_WITH from payments
    done = set()
    for e in g.of_type("LegalEntity"):
        payer_group = group_of(g, e)
        for p in g.outs(e, "PAYS"):
            payee_group = group_of(g, p.dst)
            if p.props.get("purpose") == "supplier":
                key = (p.dst, payer_group)
                if key not in done:
                    g.add_edge(p.dst, "SUPPLIER_OF", payer_group, inferred_from="PAYS(purpose=supplier)")
                    done.add(key)
                    counts["SUPPLIER_OF"] += 1
            if payee_group and payee_group != payer_group and (payer_group, payee_group) not in done:
                g.add_edge(payer_group, "TRADES_WITH", payee_group, inferred_from="PAYS")
                done.add((payer_group, payee_group))
                counts["TRADES_WITH"] += 1
    return dict(counts)


@lru_cache(maxsize=1)
def load_graph() -> tuple[KnowledgeGraph, dict]:
    return build_graph()


if __name__ == "__main__":
    _g, rep = build_graph()
    print(json.dumps(rep, indent=2, default=str))
