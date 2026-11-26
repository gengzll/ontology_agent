"""Shared CIB client ontology.

The ontology is the contract between source systems, the knowledge graph and the
agent: every node / edge loaded into the graph must use a class / relation that
is declared here, and the agent's tools are described in these terms.

It contains three kinds of knowledge:
  1. Schema      - classes, datatype properties, object relations (domain/range).
  2. Axioms      - inference rules the reasoner materialises (transitivity,
                   derived relations such as SUPPLIER_OF from payment purpose).
  3. Playbooks   - domain knowledge linking CorporateEvent types / signals to
                   product needs (GB / TB / Markets), with context conditions.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. Classes
# ---------------------------------------------------------------------------
CLASSES: dict[str, dict] = {
    "Organization": {"parent": None, "desc": "Any company or legal person."},
    "ClientGroup": {"parent": "Organization",
                    "desc": "Ultimate-parent level client relationship (the unit bankers cover).",
                    "props": ["name", "industry", "hq_country", "functional_ccy", "segment", "risk_rating"]},
    "LegalEntity": {"parent": "Organization",
                    "desc": "A booking legal entity that belongs to a ClientGroup.",
                    "props": ["name", "country", "entity_type", "lei"]},
    "ExternalCompany": {"parent": "Organization",
                        "desc": "A non-client organisation (prospect, supplier, target, PE sponsor).",
                        "props": ["name", "country", "industry", "kind"]},
    "Person": {"parent": None, "desc": "A natural person."},
    "Contact": {"parent": "Person", "desc": "A client-side person (CFO, Treasurer...).",
                "props": ["name", "title"]},
    "Banker": {"parent": "Person", "desc": "An employee of the bank.",
               "props": ["name", "role", "business"]},
    "Product": {"parent": None, "desc": "A bank product; family is GB, TB or Markets.",
                "props": ["name", "family", "code"]},
    "Industry": {"parent": None, "desc": "Industry classification.", "props": ["name"]},
    "Country": {"parent": None, "desc": "Jurisdiction.", "props": ["name", "currency"]},
    "Currency": {"parent": None, "desc": "ISO currency.", "props": ["code"]},
    "WalletEstimate": {"parent": None, "desc": "Estimated annual fee pool of a client for a product family.",
                       "props": ["family", "year", "wallet_usd", "our_revenue_usd", "share"]},
    "Opportunity": {"parent": None, "desc": "Pipeline opportunity (CRM).",
                    "props": ["stage", "est_revenue_usd", "created", "updated", "status"]},
    "Deal": {"parent": None, "desc": "Executed historical deal / mandate.",
             "props": ["date", "size_usd", "revenue_usd", "summary"]},
    "CorporateEvent": {"parent": None, "desc": "Corporate event / news about a client.",
                       "props": ["event_type", "date", "headline", "amount_usd"]},
    "Document": {"parent": None, "desc": "Unstructured document.", "props": ["title", "date", "text"]},
    "CallReport": {"parent": "Document", "desc": "Banker call report from a client meeting."},
    "CreditPaper": {"parent": "Document", "desc": "Annual credit review paper."},
    "Interaction": {"parent": None, "desc": "Banker-contact touchpoint (meeting, call, email).",
                    "props": ["channel", "date"]},
    "Intent": {"parent": None, "desc": "Client intent extracted from text (e.g. 'expand_to_VN').",
               "props": ["kind", "polarity"]},
}

# ---------------------------------------------------------------------------
# 2. Relations  name -> (domain classes, range classes, description, edge props)
# ---------------------------------------------------------------------------
RELATIONS: dict[str, dict] = {
    "BELONGS_TO_GROUP": {"domain": ["LegalEntity"], "range": ["ClientGroup"], "desc": "Entity is part of group."},
    "SUBSIDIARY_OF": {"domain": ["LegalEntity"], "range": ["LegalEntity"], "desc": "Direct ownership.",
                      "props": ["ownership_pct"], "transitive": True},
    "INCORPORATED_IN": {"domain": ["LegalEntity", "ExternalCompany"], "range": ["Country"], "desc": "Jurisdiction."},
    "IN_INDUSTRY": {"domain": ["ClientGroup", "ExternalCompany"], "range": ["Industry"], "desc": "Industry."},
    "HAS_CURRENCY": {"domain": ["Country"], "range": ["Currency"], "desc": "Local currency."},
    "HOLDS": {"domain": ["LegalEntity"], "range": ["Product"], "desc": "Product holding with the bank.",
              "props": ["since", "limit_usd", "balance_usd"]},
    "PAYS": {"domain": ["LegalEntity"], "range": ["LegalEntity", "ExternalCompany"],
             "desc": "Aggregated payment flow (12 months).",
             "props": ["currency", "annual_usd", "purpose", "monthly_usd", "rail"]},
    "COVERS": {"domain": ["Banker"], "range": ["ClientGroup"], "desc": "Coverage assignment.", "props": ["role"]},
    "WORKS_AT": {"domain": ["Contact"], "range": ["LegalEntity", "ExternalCompany"], "desc": "Current employer.",
                 "props": ["title"]},
    "PREVIOUSLY_WORKED_AT": {"domain": ["Contact"], "range": ["LegalEntity", "ExternalCompany"],
                             "desc": "Former employer.", "props": ["title", "until"]},
    "DIRECTOR_OF": {"domain": ["Contact"], "range": ["LegalEntity", "ExternalCompany"], "desc": "Board seat."},
    "JV_PARTNER_WITH": {"domain": ["LegalEntity", "ExternalCompany"], "range": ["LegalEntity", "ExternalCompany"],
                        "desc": "Joint venture partnership.", "props": ["jv_name"]},
    "HAS_WALLET": {"domain": ["ClientGroup"], "range": ["WalletEstimate"], "desc": "Wallet estimate."},
    "HAS_OPPORTUNITY": {"domain": ["ClientGroup"], "range": ["Opportunity"], "desc": "Pipeline item."},
    "FOR_PRODUCT": {"domain": ["Opportunity", "Deal"], "range": ["Product"], "desc": "Product of opp / deal."},
    "OWNED_BY": {"domain": ["Opportunity"], "range": ["Banker"], "desc": "Opportunity owner."},
    "EXECUTED_FOR": {"domain": ["Deal"], "range": ["LegalEntity"], "desc": "Deal client entity."},
    "DEAL_TEAM": {"domain": ["Deal"], "range": ["Banker"], "desc": "Deal team member."},
    "DEAL_COUNTERPARTY": {"domain": ["Deal"], "range": ["LegalEntity", "ExternalCompany"],
                          "desc": "Counterparty / target in the deal."},
    "ORIGINATED_FROM": {"domain": ["Deal", "Opportunity"], "range": ["CorporateEvent"], "desc": "Event lineage."},
    "AFFECTS": {"domain": ["CorporateEvent"], "range": ["ClientGroup", "LegalEntity"], "desc": "Event subject."},
    "INVOLVES": {"domain": ["CorporateEvent"], "range": ["LegalEntity", "ExternalCompany"],
                 "desc": "Other party (target, buyer...)."},
    "IN_COUNTRY": {"domain": ["CorporateEvent"], "range": ["Country"], "desc": "Event geography."},
    "ABOUT": {"domain": ["Document", "Interaction"], "range": ["ClientGroup"], "desc": "Document subject."},
    "AUTHORED_BY": {"domain": ["Document"], "range": ["Banker"], "desc": "Author."},
    "ATTENDEE": {"domain": ["CallReport"], "range": ["Contact"], "desc": "Client attendee."},
    "MENTIONS": {"domain": ["Document"], "range": ["Product", "Country", "Organization", "Intent"],
                 "desc": "Entity / intent mentioned in text (extracted)."},
    "WITH_CONTACT": {"domain": ["Interaction"], "range": ["Contact"], "desc": "Client side of touchpoint."},
    "BY_BANKER": {"domain": ["Interaction"], "range": ["Banker"], "desc": "Bank side of touchpoint."},
    "EXPRESSES": {"domain": ["CallReport"], "range": ["Intent"], "desc": "Client intent captured in a meeting."},
    # ---- inferred by the reasoner (never loaded directly) ----
    "SUPPLIER_OF": {"domain": ["LegalEntity", "ExternalCompany"], "range": ["ClientGroup"],
                    "desc": "INFERRED: payee of supplier payments from a group entity.", "inferred": True},
    "ULTIMATE_PARENT": {"domain": ["LegalEntity"], "range": ["LegalEntity"],
                        "desc": "INFERRED: transitive closure of SUBSIDIARY_OF.", "inferred": True},
    "TRADES_WITH": {"domain": ["ClientGroup"], "range": ["ClientGroup"],
                    "desc": "INFERRED: two client groups exchange payments.", "inferred": True},
}

# ---------------------------------------------------------------------------
# 3. Product catalogue (reference data that is part of the ontology)
# ---------------------------------------------------------------------------
PRODUCTS: list[dict] = [
    # Transaction Banking
    {"code": "CASH_MGMT", "name": "Cash Management & Operating Accounts", "family": "TB"},
    {"code": "LIQUIDITY", "name": "Liquidity Management / Cash Pooling", "family": "TB"},
    {"code": "VIRTUAL_ACCT", "name": "Virtual Accounts", "family": "TB"},
    {"code": "TRADE_FIN", "name": "Trade Finance (LC / Guarantees)", "family": "TB"},
    {"code": "SCF", "name": "Supply Chain Finance", "family": "TB"},
    # Markets
    {"code": "FX_HEDGE", "name": "FX Forwards & Hedging", "family": "Markets"},
    {"code": "IRS_HEDGE", "name": "Interest Rate Swaps", "family": "Markets"},
    {"code": "COMMODITY_HEDGE", "name": "Commodity Hedging", "family": "Markets"},
    {"code": "CUSTODY", "name": "Custody & Securities Services", "family": "Markets"},
    # Global Banking
    {"code": "RCF", "name": "Revolving Credit Facility", "family": "GB"},
    {"code": "TERM_LOAN", "name": "Term Loan / Project Finance", "family": "GB"},
    {"code": "DCM", "name": "Debt Capital Markets (Bonds)", "family": "GB"},
    {"code": "ECM", "name": "Equity Capital Markets (IPO / Placement)", "family": "GB"},
    {"code": "MNA", "name": "M&A Advisory", "family": "GB"},
    {"code": "ACQ_FIN", "name": "Acquisition Finance", "family": "GB"},
]
PRODUCT_BY_CODE = {p["code"]: p for p in PRODUCTS}

COUNTRIES: dict[str, dict] = {
    "HK": {"name": "Hong Kong", "currency": "HKD"}, "SG": {"name": "Singapore", "currency": "SGD"},
    "CN": {"name": "China", "currency": "CNY"}, "IN": {"name": "India", "currency": "INR"},
    "VN": {"name": "Vietnam", "currency": "VND"}, "ID": {"name": "Indonesia", "currency": "IDR"},
    "MY": {"name": "Malaysia", "currency": "MYR"}, "TH": {"name": "Thailand", "currency": "THB"},
    "AE": {"name": "United Arab Emirates", "currency": "AED"}, "GB": {"name": "United Kingdom", "currency": "GBP"},
    "DE": {"name": "Germany", "currency": "EUR"}, "US": {"name": "United States", "currency": "USD"},
    "JP": {"name": "Japan", "currency": "JPY"}, "AU": {"name": "Australia", "currency": "AUD"},
    "MX": {"name": "Mexico", "currency": "MXN"},
}

# ---------------------------------------------------------------------------
# 4. Playbooks: CorporateEvent type -> conditional product needs
#    Each rule: product code + condition name evaluated by engines/analytics.py
#    against the graph context of the event.
# ---------------------------------------------------------------------------
EVENT_PLAYBOOK: dict[str, list[dict]] = {
    "ACQUISITION": [
        {"product": "MNA", "condition": "no_recent_mna_mandate", "why": "Buy-side advisory on the acquisition"},
        {"product": "ACQ_FIN", "condition": "leverage_headroom", "why": "Leverage headroom (<3.5x) to fund the deal"},
        {"product": "FX_HEDGE", "condition": "cross_currency", "why": "Target currency differs from functional currency"},
        {"product": "CASH_MGMT", "condition": "no_tb_presence_in_country", "why": "No TB footprint in target country"},
    ],
    "BOND_MATURITY": [
        {"product": "DCM", "condition": "investment_grade", "why": "IG issuer can refinance in bond market"},
        {"product": "TERM_LOAN", "condition": "sub_investment_grade", "why": "Sub-IG: refinance via loan market"},
        {"product": "IRS_HEDGE", "condition": "no_rates_hedge", "why": "New debt with no rates hedging in place"},
    ],
    "MARKET_EXPANSION": [
        {"product": "CASH_MGMT", "condition": "no_tb_presence_in_country", "why": "New country without TB coverage"},
        {"product": "TRADE_FIN", "condition": "trade_intensive_industry", "why": "Trade-intensive industry entering new market"},
        {"product": "FX_HEDGE", "condition": "cross_currency", "why": "New currency exposure"},
    ],
    "IPO_PLAN": [
        {"product": "ECM", "condition": "always", "why": "Equity listing"},
        {"product": "LIQUIDITY", "condition": "no_liquidity_product", "why": "Deploy IPO proceeds"},
        {"product": "CUSTODY", "condition": "always", "why": "Share registry / custody"},
    ],
    "CAPEX_PROJECT": [
        {"product": "TERM_LOAN", "condition": "leverage_headroom", "why": "Project funding within leverage headroom"},
        {"product": "IRS_HEDGE", "condition": "no_rates_hedge", "why": "Hedge floating-rate project debt"},
        {"product": "COMMODITY_HEDGE", "condition": "commodity_intensive_industry", "why": "Commodity input exposure"},
    ],
    "SUPPLY_CHAIN_SHIFT": [
        {"product": "SCF", "condition": "no_scf_product", "why": "Onboard new supplier base onto SCF"},
        {"product": "TRADE_FIN", "condition": "trade_intensive_industry", "why": "New import corridor"},
        {"product": "FX_HEDGE", "condition": "cross_currency", "why": "Supplier invoices in new currency"},
    ],
    "NEW_SUBSIDIARY": [
        {"product": "CASH_MGMT", "condition": "no_tb_presence_in_country", "why": "New entity needs operating accounts"},
        {"product": "VIRTUAL_ACCT", "condition": "has_cash_mgmt_elsewhere", "why": "Extend group VA structure"},
    ],
    "DIVESTMENT": [
        {"product": "MNA", "condition": "no_recent_mna_mandate", "why": "Sell-side advisory"},
        {"product": "LIQUIDITY", "condition": "no_liquidity_product", "why": "Deploy disposal proceeds"},
    ],
    "MGMT_CHANGE": [
        {"product": None, "condition": "always", "why": "Relationship action: brief and meet the new executive"},
    ],
    "RATING_DOWNGRADE": [
        {"product": None, "condition": "always", "why": "Risk action: review limits; suppress new leverage asks"},
    ],
}

# Signal -> product rules used by the cross-sell / NBA engine.
SIGNAL_RULES: list[dict] = [
    {"signal": "UNHEDGED_FX_FLOWS", "product": "FX_HEDGE", "weight": 0.35,
     "desc": "Material payment flows in non-functional currencies and no FX hedging with us"},
    {"signal": "SUPPLIER_CONCENTRATION", "product": "SCF", "weight": 0.30,
     "desc": "Large recurring supplier payments and no Supply Chain Finance programme"},
    {"signal": "MULTI_COUNTRY_NO_POOLING", "product": "LIQUIDITY", "weight": 0.30,
     "desc": "Operating accounts in >=3 countries but no cash pooling"},
    {"signal": "SUBSIDIARY_BANKS_ELSEWHERE", "product": "CASH_MGMT", "weight": 0.30,
     "desc": "Subsidiary pays/receives via another bank (competitor rail)"},
    {"signal": "FLOATING_DEBT_UNHEDGED", "product": "IRS_HEDGE", "weight": 0.25,
     "desc": "Term debt with us and no interest-rate hedge"},
    {"signal": "WALLET_GAP", "product": None, "weight": 0.20,
     "desc": "Large wallet in a family with low share-of-wallet"},
    {"signal": "CALL_REPORT_INTENT", "product": None, "weight": 0.40,
     "desc": "Client expressed a relevant intent in a recent meeting"},
    {"signal": "PEER_ADOPTION", "product": None, "weight": 0.10,
     "desc": "Most industry peers hold this product"},
]

TRADE_INTENSIVE = {"Logistics", "Textiles & Apparel", "Consumer Goods", "Electronics", "Agri-Commodities",
                   "Automotive", "Chemicals", "Food & Beverage", "Shipping"}
COMMODITY_INTENSIVE = {"Steel & Metals", "Mining", "Chemicals", "Agri-Commodities", "Airlines",
                       "Petrochemicals", "Cement & Building Materials", "Paper & Packaging"}
INVESTMENT_GRADE = {"A", "A-", "BBB+", "BBB", "BBB-"}


def validate_edge(rel: str, src_type: str, dst_type: str) -> bool:
    """Check a triple against the ontology (domain/range, respecting subclassing)."""
    spec = RELATIONS.get(rel)
    if spec is None:
        return False
    return _is_a(src_type, spec["domain"]) and _is_a(dst_type, spec["range"])


def _is_a(cls: str, allowed: list[str]) -> bool:
    while cls:
        if cls in allowed:
            return True
        cls = CLASSES.get(cls, {}).get("parent")
    return False


def to_turtle() -> str:
    """Export the schema as OWL/RDFS Turtle so it can be loaded into a triple store."""
    lines = [
        "@prefix cib: <https://example.org/cib-ontology#> .",
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "",
        "<https://example.org/cib-ontology> a owl:Ontology ;",
        '    rdfs:label "CIB Frontline Client Intelligence Ontology (demo)" .',
        "",
    ]
    for name, c in CLASSES.items():
        lines.append(f"cib:{name} a owl:Class ;")
        if c.get("parent"):
            lines.append(f"    rdfs:subClassOf cib:{c['parent']} ;")
        lines.append(f'    rdfs:comment "{c["desc"]}" .')
    lines.append("")
    for name, r in RELATIONS.items():
        kind = "owl:TransitiveProperty" if r.get("transitive") else "owl:ObjectProperty"
        lines.append(f"cib:{name} a {kind} ;")
        dom = " ".join(f"cib:{d}" for d in r["domain"])
        rng = " ".join(f"cib:{d}" for d in r["range"])
        lines.append(f"    rdfs:domain [ owl:unionOf ( {dom} ) ] ;")
        lines.append(f"    rdfs:range [ owl:unionOf ( {rng} ) ] ;")
        lines.append(f'    rdfs:comment "{r["desc"]}" .')
    return "\n".join(lines) + "\n"


def summary() -> dict:
    return {
        "classes": {k: {"parent": v.get("parent"), "desc": v["desc"], "props": v.get("props", [])}
                    for k, v in CLASSES.items()},
        "relations": {k: {"domain": v["domain"], "range": v["range"], "desc": v["desc"],
                          "inferred": v.get("inferred", False)} for k, v in RELATIONS.items()},
        "products": PRODUCTS,
        "event_playbook": EVENT_PLAYBOOK,
        "signal_rules": SIGNAL_RULES,
    }
