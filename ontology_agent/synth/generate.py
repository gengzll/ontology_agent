"""Synthetic data generator for the Frontline Client Intelligence Graph demo.

Produces data the way it would arrive from *separate* source systems (CRM,
payments, product, wallet, finance, deals, news, documents) - deliberately with
inconsistent naming so that the knowledge-graph build has to do entity
resolution.  All companies and people are fictional.

It also writes a hidden `ground_truth/` folder that only the experiment harness
reads (realised opportunities, event outcomes, gold answers).  The agent never
sees it.

Run:  python -m ontology_agent.synth.generate
"""

from __future__ import annotations

import csv
import json
import random
import shutil
from datetime import date, timedelta

from .. import AS_OF, DATA_DIR, SOURCE_DIR
from ..ontology import (COMMODITY_INTENSIVE, COUNTRIES, EVENT_PLAYBOOK, INVESTMENT_GRADE, PRODUCT_BY_CODE,
                        TRADE_INTENSIVE)

SEED = 20261126
TODAY = date.fromisoformat(AS_OF)
GT_DIR = DATA_DIR / "ground_truth"

# ---------------------------------------------------------------------------
# Reference lists (all fictional)
# ---------------------------------------------------------------------------
GROUPS = [  # name, industry, hq
    ("Aurora Logistics", "Logistics", "SG"),
    ("Banyan Foods", "Food & Beverage", "MY"),
    ("Cobalt Semicon", "Electronics", "SG"),
    ("Dragonfly Renewables", "Utilities & Renewables", "HK"),
    ("Evergreen Textiles", "Textiles & Apparel", "HK"),
    ("Falcon Petrochem", "Petrochemicals", "AE"),
    ("Granite Infrastructure", "Construction & Infrastructure", "IN"),
    ("Harbourline Retail", "Consumer Goods", "HK"),
    ("Ironwood Pharma", "Healthcare & Pharma", "IN"),
    ("Jadeport Maritime", "Shipping", "SG"),
    ("Kestrel Auto Components", "Automotive", "CN"),
    ("Lotusbay Hospitality", "Hospitality", "TH"),
    ("Meridale Telecom", "Telecom & Media", "ID"),
    ("Nimbusline Cloud", "Technology", "SG"),
    ("Orchidfield Agri", "Agri-Commodities", "SG"),
    ("Pinewood Steelworks", "Steel & Metals", "IN"),
    ("Quartzline Electronics", "Electronics", "CN"),
    ("Redcliff Packaging", "Paper & Packaging", "MY"),
    ("Saffronleaf Brands", "Consumer Goods", "IN"),
    ("Tidewell Shipping", "Shipping", "HK"),
    ("Umberstone Mining", "Mining", "AU"),
    ("Vantagepoint Healthcare", "Healthcare & Pharma", "SG"),
    ("Willowmere Education", "Education", "GB"),
    ("Xenara Battery Tech", "Automotive", "CN"),
    ("Yarrowfield Chemicals", "Chemicals", "DE"),
    ("Zephyrine Airways", "Airlines", "HK"),
    ("Atlasrock Cement", "Cement & Building Materials", "VN"),
    ("Beaconhill Insurance Services", "Financial Services", "GB"),
    ("Cedarvale Properties", "Real Estate", "SG"),
    ("Deltawave Apparel Sourcing", "Textiles & Apparel", "HK"),
]
N_COMPLEX = 15          # first 15 groups have deep multi-country structures
N_NBA = 25              # first 25 groups are the cross-sell experiment cohort

EXTERNALS = [  # name, country, industry, kind
    ("Helixon Components", "CN", "Electronics", "supplier"),
    ("Monsoon Freight Lines", "IN", "Logistics", "supplier"),
    ("Polarion Mining Equipment", "AU", "Industrial Machinery", "supplier"),
    ("Sakuraden Precision", "JP", "Industrial Machinery", "supplier"),
    ("Verdant Packaging Co", "VN", "Paper & Packaging", "supplier"),
    ("Northstar Harbour Capital", "SG", "Private Equity", "sponsor"),
    ("Bluefin Seafood Trading", "TH", "Food & Beverage", "supplier"),
    ("Corvid Analytics", "GB", "Technology", "target"),
    ("Riverside Polymers", "MY", "Chemicals", "supplier"),
    ("Saltmarsh Agro", "ID", "Agri-Commodities", "supplier"),
    ("Tessellate Robotics", "DE", "Industrial Machinery", "target"),
    ("Orbiton Satellite", "US", "Telecom & Media", "target"),
    ("Granary Hill Mills", "AU", "Agri-Commodities", "target"),
    ("Kinetica Motors", "MX", "Automotive", "target"),
    ("Lumen Grid Solutions", "VN", "Utilities & Renewables", "target"),
    ("Pacific Rim Cold Chain", "VN", "Logistics", "target"),
    ("Silverline Textiles Mill", "IN", "Textiles & Apparel", "supplier"),
    ("Ember Coal Trading", "ID", "Mining", "supplier"),
    ("Coralbay Resorts", "ID", "Hospitality", "target"),
    ("Quantafin Software", "IN", "Technology", "supplier"),
    ("Harrowgate Steel Trading", "AE", "Steel & Metals", "supplier"),
    ("Mistral Chemicals Asia", "SG", "Chemicals", "supplier"),
]

FIRST = ["Wei", "Priya", "Daniel", "Mei Ling", "Arjun", "Siti", "Kenji", "Olivia", "Rahul", "Nguyen", "Hannah",
         "Farah", "Marcus", "Yuki", "Ahmed", "Chloe", "Ravi", "Lina", "Tom", "Ananya", "Jun", "Grace", "Omar",
         "Elena", "Ken", "Aisha", "Lucas", "Nadia", "Vikram", "Sophie", "Hiro", "Isabel", "Kai", "Maya", "Ivan",
         "Leila", "Ben", "Rina", "Felix", "Zara"]
LAST = ["Tan", "Sharma", "Lee", "Wong", "Patel", "Rahman", "Sato", "Brown", "Gupta", "Tran", "Mueller", "Hassan",
        "Lim", "Nakamura", "Khan", "Martin", "Iyer", "Chen", "Walker", "Reddy", "Ho", "Kaur", "Santos",
        "Fischer", "Ong", "Suzuki", "Silva", "Nair", "Yeo", "Kim"]

BANKERS = [  # id, name, role, business
    ("BK01", "Sarah Lim", "Relationship Manager", "Coverage"),
    ("BK02", "James Okafor", "Relationship Manager", "Coverage"),
    ("BK03", "Mei Chen", "Relationship Manager", "Coverage"),
    ("BK04", "Rohan Mehta", "Relationship Manager", "Coverage"),
    ("BK05", "Anna Schmidt", "Senior Banker", "Coverage"),
    ("BK06", "David Ng", "M&A Director", "GB"),
    ("BK07", "Laura Pereira", "DCM / LevFin Director", "GB"),
    ("BK08", "Kevin Tan", "Cash Management Specialist", "TB"),
    ("BK09", "Fatima Al-Sayed", "Trade & SCF Specialist", "TB"),
    ("BK10", "Hiroshi Mori", "FX & Rates Sales", "Markets"),
    ("BK11", "Emily Zhou", "Commodities Sales", "Markets"),
    ("BK12", "Arun Pillai", "Corporate Finance Analyst", "GB"),
]
SPECIALIST_BY_FAMILY = {"GB": ["BK06", "BK07", "BK12"], "TB": ["BK08", "BK09"], "Markets": ["BK10", "BK11"]}

ENTITY_SUFFIX = {
    "SG": "(Singapore) Pte Ltd", "HK": "(Hong Kong) Limited", "CN": "(Shanghai) Co., Ltd", "IN": "India Private Limited",
    "VN": "Vietnam Co., Ltd", "ID": "Indonesia", "MY": "(M) Sdn Bhd", "TH": "(Thailand) Co., Ltd", "AE": "FZE",
    "GB": "UK Limited", "DE": "GmbH", "US": "Inc.", "JP": "KK", "AU": "Pty Ltd", "MX": "de Mexico SA de CV",
}
PAYMENT_ALIAS_COUNTRY = {"SG": "(S)", "MY": "(M)", "HK": "HK", "TH": "(THAILAND)", "IN": "INDIA PVT"}


def _entity_name(short: str, cc: str) -> str:
    if cc == "ID":
        return f"PT {short} Indonesia"
    return f"{short} {ENTITY_SUFFIX[cc]}"


class Gen:
    def __init__(self, seed: int = SEED):
        self.r = random.Random(seed)
        self.groups: list[dict] = []
        self.entities: list[dict] = []
        self.externals: list[dict] = []
        self.contacts: list[dict] = []
        self.coverage: list[dict] = []
        self.holdings: list[dict] = []
        self.payments: list[dict] = []
        self.wallet: list[dict] = []
        self.financials: list[dict] = []
        self.pipeline: list[dict] = []
        self.deals: list[dict] = []
        self.events: list[dict] = []
        self.interactions: list[dict] = []
        self.call_reports: list[dict] = []
        self.credit_papers: list[dict] = []
        self.jvs: list[dict] = []
        # hidden ground truth
        self.gt_nba: list[dict] = []
        self.gt_events: list[dict] = []
        self.gt_rel: list[dict] = []
        self.gt_aliases: list[dict] = []
        self._used_names: set[str] = set()
        self._call_ctr = 0
        self._int_ctr = 0
        self._deal_ctr = 0
        self._opp_ctr = 0
        self._months = [self._month_add(TODAY.replace(day=1), -i) for i in range(12, 0, -1)]

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _month_add(d: date, n: int) -> date:
        y, m = d.year, d.month + n
        while m <= 0:
            m += 12
            y -= 1
        while m > 12:
            m -= 12
            y += 1
        return date(y, m, 1)

    def _person(self) -> str:
        while True:
            n = f"{self.r.choice(FIRST)} {self.r.choice(LAST)}"
            if n not in self._used_names:
                self._used_names.add(n)
                return n

    def _rand_date(self, start: date, end: date) -> date:
        return start + timedelta(days=self.r.randint(0, max(0, (end - start).days)))

    def ents_of(self, gid: str) -> list[dict]:
        return [e for e in self.entities if e["group_id"] == gid]

    def holds(self, eid: str, code: str, at: date | None = None) -> bool:
        return any(h["entity_id"] == eid and h["product_code"] == code and
                   (at is None or date.fromisoformat(h["since"]) <= at) for h in self.holdings)

    def group_holds(self, gid: str, code: str, at: date | None = None) -> bool:
        return any(self.holds(e["entity_id"], code, at) for e in self.ents_of(gid))

    def add_holding(self, eid: str, code: str, since: date, limit: float = 0.0, balance: float = 0.0):
        if self.holds(eid, code):
            return
        self.holdings.append({"entity_id": eid, "product_code": code, "since": since.isoformat(),
                              "limit_usd": round(limit), "balance_usd": round(balance)})

    def group(self, gid: str) -> dict:
        return next(g for g in self.groups if g["group_id"] == gid)

    # ------------------------------------------------------- 1. structure
    def build_groups(self):
        ratings = ["A", "A-", "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-"]
        for i, (name, ind, hq) in enumerate(GROUPS):
            gid = f"G{i + 1:03d}"
            self.groups.append({
                "group_id": gid, "name": f"{name} Group", "short": name, "industry": ind, "hq_country": hq,
                "functional_ccy": COUNTRIES[hq]["currency"] if hq not in ("HK", "AE") else "USD",
                "segment": "Multinational" if i < N_COMPLEX else "Large Corporate",
                "risk_rating": self.r.choice(ratings[:5] if i % 3 else ratings),
            })
        for j, (name, cc, ind, kind) in enumerate(EXTERNALS):
            self.externals.append({"ext_id": f"X{j + 1:03d}", "name": name, "country": cc, "industry": ind,
                                   "kind": kind})

    def build_entities(self):
        all_cc = [c for c in COUNTRIES]
        for i, g in enumerate(self.groups):
            short, hq = g["short"], g["hq_country"]
            parent_name = f"{short} Holdings Limited" if hq in ("HK", "SG", "GB", "AE") else _entity_name(short, hq)
            parent = self._new_entity(g, parent_name, hq, None, 100, "Ultimate Parent")
            if i < N_COMPLEX:
                # regional holdco + 3-6 operating companies, some under the holdco (multi-level)
                rh_cc = "SG" if hq != "SG" else "HK"
                rh = self._new_entity(g, f"{short} Asia Holdings {ENTITY_SUFFIX[rh_cc]}", rh_cc, parent, 100,
                                      "Regional Holdco")
                n = self.r.randint(3, 6)
                ccs = self.r.sample([c for c in all_cc if c not in (hq, rh_cc)], n)
                for k, cc in enumerate(ccs):
                    par = rh if (k % 2 == 0 and cc not in ("GB", "DE", "US", "MX")) else parent
                    pct = self.r.choice([100, 100, 100, 75, 60, 51])
                    self._new_entity(g, _entity_name(short, cc), cc, par, pct, "Operating Company")
            else:
                for cc in self.r.sample([c for c in all_cc if c != hq], self.r.randint(1, 2)):
                    self._new_entity(g, _entity_name(short, cc), cc, parent, 100, "Operating Company")

    def _new_entity(self, g, name, cc, parent, pct, etype) -> dict:
        eid = f"E{len(self.entities) + 1:04d}"
        e = {"entity_id": eid, "group_id": g["group_id"], "name": name, "country": cc,
             "parent_entity_id": parent["entity_id"] if parent else "", "ownership_pct": pct,
             "entity_type": etype, "lei": f"5493{self.r.randint(10**15, 10**16 - 1)}"}
        self.entities.append(e)
        return e

    def build_people(self):
        for g in self.groups:
            ents = self.ents_of(g["group_id"])
            parent = ents[0]
            for title in ["Group CFO", "Group Treasurer", "CEO"] + (["Head of Procurement"] if self.r.random() < .6
                                                                    else []):
                self.contacts.append({"contact_id": f"C{len(self.contacts) + 1:04d}", "name": self._person(),
                                      "title": title, "employer_id": parent["entity_id"],
                                      "employer_name": parent["name"], "prior_employer_name": "", "prior_title": "",
                                      "prior_until": "", "board_seats": ""})
            for e in ents[2:4]:  # local finance directors at a couple of subsidiaries
                self.contacts.append({"contact_id": f"C{len(self.contacts) + 1:04d}", "name": self._person(),
                                      "title": f"Finance Director, {COUNTRIES[e['country']]['name']}",
                                      "employer_id": e["entity_id"], "employer_name": e["name"],
                                      "prior_employer_name": "", "prior_title": "", "prior_until": "",
                                      "board_seats": ""})
        for x in self.externals:
            for title in ["CFO", "CEO"]:
                self.contacts.append({"contact_id": f"C{len(self.contacts) + 1:04d}", "name": self._person(),
                                      "title": title, "employer_id": x["ext_id"], "employer_name": x["name"],
                                      "prior_employer_name": "", "prior_title": "", "prior_until": "",
                                      "board_seats": ""})
        # coverage
        rms = ["BK01", "BK02", "BK03", "BK04", "BK05"]
        for i, g in enumerate(self.groups):
            self.coverage.append({"banker_id": rms[i % 5], "group_id": g["group_id"], "role": "Lead RM"})
            for fam, ids in SPECIALIST_BY_FAMILY.items():
                if self.r.random() < 0.6:
                    self.coverage.append({"banker_id": self.r.choice(ids), "group_id": g["group_id"],
                                          "role": f"{fam} Product Partner"})

    def contacts_of(self, gid: str) -> list[dict]:
        eids = {e["entity_id"] for e in self.ents_of(gid)}
        return [c for c in self.contacts if c["employer_id"] in eids]

    # ------------------------------------------------------- 2. finance & baseline holdings
    def build_financials_and_holdings(self):
        for i, g in enumerate(self.groups):
            base = self.r.uniform(800, 9000) * (1.6 if i < N_COMPLEX else 1.0)
            lev = self.r.uniform(1.0, 4.8)
            for y, growth in ((2023, 1.0), (2024, self.r.uniform(.95, 1.15)), (2025, self.r.uniform(.95, 1.2))):
                base *= growth
                ebitda = base * self.r.uniform(.09, .22)
                self.financials.append({"group_id": g["group_id"], "year": y, "revenue_usd_m": round(base, 1),
                                        "ebitda_usd_m": round(ebitda, 1),
                                        "net_debt_usd_m": round(ebitda * lev * self.r.uniform(.9, 1.1), 1)})
            ents = self.ents_of(g["group_id"])
            start = date(2019, 1, 1)
            parent = ents[0]
            self.add_holding(parent["entity_id"], "CASH_MGMT", self._rand_date(start, date(2022, 1, 1)),
                             balance=self.r.uniform(5e6, 8e7))
            self.add_holding(parent["entity_id"], "RCF", self._rand_date(start, date(2023, 1, 1)),
                             limit=self.r.choice([1e8, 1.5e8, 2.5e8, 4e8]))
            for e in ents[1:]:
                if self.r.random() < 0.55:
                    self.add_holding(e["entity_id"], "CASH_MGMT", self._rand_date(start, date(2024, 6, 1)),
                                     balance=self.r.uniform(1e6, 2e7))
            if self.r.random() < 0.45:
                self.add_holding(parent["entity_id"], "TERM_LOAN", self._rand_date(start, date(2024, 1, 1)),
                                 limit=self.r.choice([5e7, 1e8, 2e8]))
            if self.r.random() < 0.35:
                self.add_holding(self.r.choice(ents)["entity_id"], "TRADE_FIN", self._rand_date(start, date(2024, 1, 1)),
                                 limit=self.r.choice([2e7, 5e7]))

    def leverage(self, gid: str, before: date) -> float:
        rows = [f for f in self.financials if f["group_id"] == gid and f["year"] < before.year] or \
               [f for f in self.financials if f["group_id"] == gid]
        f = max(rows, key=lambda x: x["year"])
        return f["net_debt_usd_m"] / f["ebitda_usd_m"]

    # ------------------------------------------------------- 3. historical events -> deals (Experiment 2)
    def build_events(self):
        types_hist = ["ACQUISITION"] * 9 + ["BOND_MATURITY"] * 6 + ["MARKET_EXPANSION"] * 7 + ["IPO_PLAN"] * 2 + \
                     ["CAPEX_PROJECT"] * 5 + ["SUPPLY_CHAIN_SHIFT"] * 5 + ["NEW_SUBSIDIARY"] * 3 + \
                     ["DIVESTMENT"] * 2 + ["MGMT_CHANGE"] * 2 + ["RATING_DOWNGRADE"] * 1
        self.r.shuffle(types_hist)
        targets = [x for x in self.externals if x["kind"] == "target"]
        for k, et in enumerate(types_hist):
            g = self.groups[(k * 7) % len(self.groups)]
            d = self._rand_date(date(2024, 1, 15), date(2026, 3, 31))
            self._make_event(g, et, d, targets, historical=True)
        # recent events (last ~60 days) - used for "what changed in my portfolio"
        recent = [("G001", "ACQUISITION"), ("G003", "MARKET_EXPANSION"), ("G006", "RATING_DOWNGRADE"),
                  ("G011", "BOND_MATURITY"), ("G016", "MGMT_CHANGE"), ("G021", "SUPPLY_CHAIN_SHIFT"),
                  ("G008", "NEW_SUBSIDIARY"), ("G026", "CAPEX_PROJECT")]
        for gid, et in recent:
            d = self._rand_date(TODAY - timedelta(days=55), TODAY - timedelta(days=3))
            self._make_event(self.group(gid), et, d, targets, historical=False)

    def _make_event(self, g, et, d, targets, historical):
        eid = f"EV{len(self.events) + 1:03d}"
        ents = self.ents_of(g["group_id"])
        present = {e["country"] for e in ents}
        cc, cpty, amount = "", "", 0
        if et == "ACQUISITION":
            t = self.r.choice(targets)
            cc, cpty, amount = t["country"], t["name"], self.r.choice([1.5e8, 3e8, 6e8, 1.2e9])
            headline = f"{g['name']} agrees to acquire {t['name']} ({COUNTRIES[cc]['name']}) for USD {amount/1e6:,.0f}m"
        elif et == "BOND_MATURITY":
            amount = self.r.choice([3e8, 5e8, 7.5e8])
            headline = f"{g['name']}'s USD {amount/1e6:,.0f}m notes fall due in {d.year + 1}; refinancing under review"
            cc = g["hq_country"]
        elif et in ("MARKET_EXPANSION", "SUPPLY_CHAIN_SHIFT", "NEW_SUBSIDIARY"):
            options = [c for c in COUNTRIES if c not in present] if et != "SUPPLY_CHAIN_SHIFT" else list(COUNTRIES)
            if self.r.random() < 0.3 and et != "NEW_SUBSIDIARY":
                options = [c for c in present if c != g["hq_country"]] or options  # sometimes an existing country
            cc = self.r.choice(options)
            cn = COUNTRIES[cc]["name"]
            headline = {"MARKET_EXPANSION": f"{g['name']} announces entry into {cn} with new distribution hub",
                        "SUPPLY_CHAIN_SHIFT": f"{g['name']} shifts sourcing of key inputs to suppliers in {cn}",
                        "NEW_SUBSIDIARY": f"{g['name']} incorporates new subsidiary in {cn}"}[et]
            amount = self.r.choice([5e7, 1e8, 2e8])
        elif et == "IPO_PLAN":
            cc, amount = g["hq_country"], self.r.choice([4e8, 8e8])
            headline = f"{g['name']} hires advisers for potential USD {amount/1e6:,.0f}m listing of its regional unit"
        elif et == "CAPEX_PROJECT":
            cc, amount = self.r.choice(sorted(present)), self.r.choice([2e8, 4e8, 9e8])
            headline = f"{g['name']} approves USD {amount/1e6:,.0f}m capacity expansion in {COUNTRIES[cc]['name']}"
        elif et == "DIVESTMENT":
            cc, amount = self.r.choice(sorted(present)), self.r.choice([1e8, 2.5e8])
            headline = f"{g['name']} explores sale of its {COUNTRIES[cc]['name']} business"
        elif et == "MGMT_CHANGE":
            cc = g["hq_country"]
            headline = f"{g['name']} appoints new Group CFO"
        else:
            cc = g["hq_country"]
            headline = f"Rating agency downgrades {g['name']} by one notch on weaker cash flow"
        ev = {"event_id": eid, "date": d.isoformat(), "group_id": g["group_id"], "event_type": et,
              "headline": headline, "country": cc, "counterparty_name": cpty, "amount_usd": int(amount)}
        self.events.append(ev)
        if et == "MGMT_CHANGE":
            self._plant_cfo_move(g, d)
        if historical:
            self._realise_event(ev, g, d)

    def _plant_cfo_move(self, g, d):
        """New CFO joins from another client group - a relationship signal only the graph sees."""
        cfo = next(c for c in self.contacts_of(g["group_id"]) if c["title"] == "Group CFO")
        donor = self.r.choice([x for x in self.groups if x["group_id"] != g["group_id"]])
        cfo["prior_employer_name"] = self.ents_of(donor["group_id"])[0]["name"]
        cfo["prior_title"] = "Group Treasurer"
        cfo["prior_until"] = (d - timedelta(days=20)).isoformat()

    def condition(self, name: str, g: dict, ev: dict, at: date) -> bool:
        gid, cc = g["group_id"], ev["country"]
        ents = self.ents_of(gid)
        if name == "always":
            return True
        if name == "no_recent_mna_mandate":
            return not any(dl for dl in self.deals if dl["product_code"] == "MNA" and
                           self._deal_group(dl) == gid and
                           0 <= (at - date.fromisoformat(dl["date"])).days < 540)
        if name == "leverage_headroom":
            return self.leverage(gid, at) < 3.5
        if name == "cross_currency":
            return bool(cc) and COUNTRIES[cc]["currency"] != g["functional_ccy"]
        if name == "no_tb_presence_in_country":
            return not any(e["country"] == cc and self.holds(e["entity_id"], "CASH_MGMT", at) for e in ents)
        if name == "investment_grade":
            return g["risk_rating"] in INVESTMENT_GRADE
        if name == "sub_investment_grade":
            return g["risk_rating"] not in INVESTMENT_GRADE
        if name == "no_rates_hedge":
            return not self.group_holds(gid, "IRS_HEDGE", at)
        if name == "trade_intensive_industry":
            return g["industry"] in TRADE_INTENSIVE
        if name == "commodity_intensive_industry":
            return g["industry"] in COMMODITY_INTENSIVE
        if name == "no_scf_product":
            return not self.group_holds(gid, "SCF", at)
        if name == "no_liquidity_product":
            return not self.group_holds(gid, "LIQUIDITY", at)
        if name == "has_cash_mgmt_elsewhere":
            return self.group_holds(gid, "CASH_MGMT", at)
        raise KeyError(name)

    def _deal_group(self, dl) -> str:
        return next(e["group_id"] for e in self.entities if e["entity_id"] == dl["entity_id"])

    def _realise_event(self, ev, g, d):
        """Ground truth: which GB/TB/Markets opportunities actually followed the event."""
        realised = []
        for rule in EVENT_PLAYBOOK[ev["event_type"]]:
            if rule["product"] is None:
                continue
            if self.condition(rule["condition"], g, ev, d) and self.r.random() < 0.8:
                realised.append(rule["product"])
            elif not self.condition(rule["condition"], g, ev, d) and self.r.random() < 0.07:
                realised.append(rule["product"])  # rare: happened despite context
        if self.r.random() < 0.15:  # something no playbook would predict
            realised.append(self.r.choice(["TRADE_FIN", "RCF", "CUSTODY", "VIRTUAL_ACCT"]))
        for code in dict.fromkeys(realised):
            dd = d + timedelta(days=self.r.randint(30, 160))
            if dd >= TODAY:
                continue
            ents = self.ents_of(g["group_id"])
            target_ent = next((e for e in ents if e["country"] == ev["country"]), ents[0])
            self._add_deal(target_ent, code, dd, ev)
            self.gt_events.append({"event_id": ev["event_id"], "product_code": code})
        if not realised:
            self.gt_events.append({"event_id": ev["event_id"], "product_code": ""})

    def _add_deal(self, ent, code, dd, ev=None, cpty=""):
        self._deal_ctr += 1
        fam = PRODUCT_BY_CODE[code]["family"]
        size = self.r.choice([2e7, 5e7, 1e8, 2.5e8, 5e8])
        team = [next(c["banker_id"] for c in self.coverage if c["group_id"] == ent["group_id"] and
                     c["role"] == "Lead RM"), self.r.choice(SPECIALIST_BY_FAMILY[fam])]
        cp = cpty or (ev["counterparty_name"] if ev else "")
        g = self.group(ent["group_id"])
        pname = PRODUCT_BY_CODE[code]["name"]
        summary = f"{pname} for {ent['name']}"
        if ev:
            summary += f", following: {ev['headline']}"
        if cp:
            summary += f". Counterparty: {cp}"
        summary += f". Size USD {size/1e6:,.0f}m. Team: " + ", ".join(b[1] for b in BANKERS if b[0] in team) + "."
        self.deals.append({"deal_id": f"D{self._deal_ctr:04d}", "date": dd.isoformat(), "entity_id": ent["entity_id"],
                           "group_name": g["name"], "product_code": code, "size_usd": int(size),
                           "revenue_usd": int(size * self.r.uniform(.002, .012)),
                           "counterparty_name": cp, "event_id": ev["event_id"] if ev else "",
                           "team": ";".join(team), "summary": summary})
        if code not in ("MNA", "ECM"):
            self.add_holding(ent["entity_id"], code, dd, limit=size if fam == "GB" else 0)

    def build_background_deals(self):
        for g in self.groups:
            for _ in range(self.r.randint(1, 3)):
                ent = self.r.choice(self.ents_of(g["group_id"]))
                code = self.r.choice(["RCF", "TERM_LOAN", "DCM", "TRADE_FIN", "CASH_MGMT", "FX_HEDGE"])
                if code == "FX_HEDGE" and g["group_id"] in self._nba_fx_need:
                    code = "RCF"
                self._add_deal(ent, code, self._rand_date(date(2023, 6, 1), date(2026, 9, 30)))

    # ------------------------------------------------------- 4. latent needs (Experiment 1) -> data
    def plan_latent_needs(self):
        """Decide each cohort client's hidden needs *before* generating the data that hints at them."""
        self._nba_fx_need, self.latent = set(), {}
        for i, g in enumerate(self.groups[:N_NBA]):
            needs = []
            if self.r.random() < 0.45:
                needs.append(("FX_HEDGE", "UNHEDGED_FX_FLOWS"))
                self._nba_fx_need.add(g["group_id"])
            if self.r.random() < 0.40:
                needs.append(("SCF", "SUPPLIER_CONCENTRATION"))
            if self.r.random() < 0.35 and len(self.ents_of(g["group_id"])) >= 3:
                needs.append(("LIQUIDITY", "MULTI_COUNTRY_NO_POOLING"))
            if self.r.random() < 0.35 and len(self.ents_of(g["group_id"])) >= 2:
                needs.append(("CASH_MGMT", "SUBSIDIARY_BANKS_ELSEWHERE"))
            if self.r.random() < 0.40:
                needs.append((self.r.choice(["DCM", "MNA", "TRADE_FIN", "COMMODITY_HEDGE", "ECM", "IRS_HEDGE"]),
                              "CALL_REPORT_INTENT"))
            self.latent[g["group_id"]] = needs

    def build_payments(self):
        sup_pool = [x for x in self.externals if x["kind"] == "supplier"]
        for g in self.groups:
            gid = g["group_id"]
            ents = self.ents_of(gid)
            parent = ents[0]
            needs = {n[1]: n[0] for n in self.latent.get(gid, [])}
            # intercompany funding from parent to subsidiaries
            elsewhere = None
            if "SUBSIDIARY_BANKS_ELSEWHERE" in needs:
                cands = [e for e in ents[1:] if not self.holds(e["entity_id"], "CASH_MGMT")] or ents[1:]
                elsewhere = self.r.choice(cands)
                # make sure it really doesn't bank with us
                self.holdings = [h for h in self.holdings if not (h["entity_id"] == elsewhere["entity_id"]
                                                                  and h["product_code"] == "CASH_MGMT")]
            for e in ents[1:]:
                bank = "OTHER_BANK" if e is elsewhere else ("OUR_BANK" if self.holds(e["entity_id"], "CASH_MGMT")
                                                           else self.r.choice(["OUR_BANK", "OTHER_BANK"]))
                if e is not elsewhere and bank == "OTHER_BANK" and gid in self.latent:
                    bank = "OUR_BANK"  # keep the planted signal unambiguous within the cohort
                self._flow(parent, e["name"], e["country"], COUNTRIES[e["country"]]["currency"]
                           if self.r.random() < .5 else "USD", self.r.uniform(2e6, 2e7), "intercompany", bank,
                           alias_of=e)
            # suppliers
            n_sup = self.r.randint(4, 7) if "SUPPLIER_CONCENTRATION" in needs else self.r.randint(1, 2)
            scale = 3e7 if "SUPPLIER_CONCENTRATION" in needs else 3e6
            if "SUPPLIER_CONCENTRATION" in needs:
                self.holdings = [h for h in self.holdings if not (h["product_code"] == "SCF" and
                                                                  h["entity_id"] in {x["entity_id"] for x in ents})]
            for s in self.r.sample(sup_pool, min(n_sup, len(sup_pool))):
                ccy = COUNTRIES[s["country"]]["currency"] if s["country"] in COUNTRIES else "USD"
                if "UNHEDGED_FX_FLOWS" not in needs and gid in self.latent:
                    ccy = "USD" if g["functional_ccy"] != "USD" or self.r.random() < .7 else ccy
                    if ccy != g["functional_ccy"] and ccy != "USD":
                        ccy = g["functional_ccy"]
                payer = self.r.choice(ents)
                self._flow(payer, s["name"], s["country"], ccy, self.r.uniform(.5, 1.5) * scale, "supplier",
                           "EXTERNAL")
            if "UNHEDGED_FX_FLOWS" in needs:
                self.holdings = [h for h in self.holdings if not (h["product_code"] == "FX_HEDGE" and
                                                                  h["entity_id"] in {x["entity_id"] for x in ents})]
                for _ in range(2):
                    s = self.r.choice(sup_pool)
                    ccy = self.r.choice([c["currency"] for k, c in COUNTRIES.items()
                                         if c["currency"] not in (g["functional_ccy"], "USD")])
                    self._flow(self.r.choice(ents), s["name"], s["country"], ccy, self.r.uniform(2e7, 6e7),
                               "supplier", "EXTERNAL")
            if "MULTI_COUNTRY_NO_POOLING" in needs:
                self.holdings = [h for h in self.holdings if not (h["product_code"] == "LIQUIDITY" and
                                                                  h["entity_id"] in {x["entity_id"] for x in ents})]
                countries_with_cm = {e["country"] for e in ents if self.holds(e["entity_id"], "CASH_MGMT")}
                for e in ents:
                    if len(countries_with_cm) >= 3:
                        break
                    if e is not elsewhere and e["country"] not in countries_with_cm:
                        self.add_holding(e["entity_id"], "CASH_MGMT", date(2023, 3, 1), balance=5e6)
                        countries_with_cm.add(e["country"])
            elif gid in self.latent:
                # avoid accidental pooling signals in the cohort: either already pooled or < 3 countries
                cm_countries = {e["country"] for e in ents if self.holds(e["entity_id"], "CASH_MGMT")}
                if len(cm_countries) >= 3 and not self.group_holds(gid, "LIQUIDITY"):
                    self.add_holding(parent["entity_id"], "LIQUIDITY", date(2022, 5, 1))
            if gid in self.latent and "UNHEDGED_FX_FLOWS" not in needs and self.r.random() < .5:
                self.add_holding(parent["entity_id"], "FX_HEDGE", date(2021, 9, 1))
            # customer receipts / dividends to parent
            self._flow(parent, f"{g['short']} Shareholders", g["hq_country"], g["functional_ccy"],
                       self.r.uniform(5e6, 3e7), "dividend", "EXTERNAL")
        self._plant_client_to_client_flows()
        self._plant_attrition()

    def _plant_client_to_client_flows(self):
        """Some clients pay each other - an inter-client supply chain only visible in payments."""
        pairs = [("G002", "G018"), ("G005", "G030"), ("G017", "G003"), ("G024", "G011"), ("G014", "G013"),
                 ("G015", "G002"), ("G010", "G001"), ("G007", "G016")]
        for a, b in pairs:
            payer = self.r.choice(self.ents_of(a))
            payee = self.r.choice(self.ents_of(b))
            self._flow(payer, payee["name"], payee["country"], "USD", self.r.uniform(8e6, 4e7), "supplier",
                       "OUR_BANK", alias_of=payee)

    def _plant_attrition(self):
        """Flows collapsing in the last two months -> possible share loss to a competitor."""
        self.attrition = {"G002": 0.35, "G013": 0.45}
        self.spike = {"G019": 2.2}
        for p in self.payments:
            gid = p["_gid"]
            m = p["month"]
            if gid in self.attrition and m >= self._months[-2].isoformat()[:7]:
                p["amount_usd"] = int(p["amount_usd"] * self.attrition[gid])
            if gid in self.spike and m == self._months[-1].isoformat()[:7]:
                p["amount_usd"] = int(p["amount_usd"] * self.spike[gid])

    def _alias(self, name: str, cc: str) -> str:
        """Payment systems store free-text beneficiary names."""
        roll = self.r.random()
        up = name.upper().replace(",", "").replace(".", "")
        if roll < 0.40:
            return name
        if roll < 0.73:
            return up
        if roll < 0.90 and "(" in name:
            short = up.split(" (")[0]
            return f"{short} {PAYMENT_ALIAS_COUNTRY.get(cc, cc)} {up.split(') ')[-1]}".strip()
        if roll < 0.97:
            # abbreviated tokens - defeats a naive exact-match resolver
            toks = [t[:3] for t in up.replace("(", "").replace(")", "").split()[:2]]
            return " ".join(toks) + f" {cc}"
        # garbled free text - realistically unresolvable
        initials = "".join(t[0] for t in up.split()[:3])
        return f"{initials} TREASURY ACCT REF {self.r.randint(1000, 9999)}"

    def _flow(self, payer, payee_name, payee_cc, ccy, annual, purpose, bank, alias_of=None):
        gid = payer["group_id"]
        for m in self._months:
            amt = annual / 12 * self.r.uniform(.75, 1.25)
            raw = self._alias(payee_name, payee_cc) if alias_of is not None else payee_name
            if alias_of is not None:
                self.gt_aliases.append({"raw_name": raw, "entity_id": alias_of["entity_id"]})
            self.payments.append({"month": m.isoformat()[:7], "payer_entity_id": payer["entity_id"],
                                  "beneficiary_name": raw, "beneficiary_country": payee_cc, "currency": ccy,
                                  "amount_usd": int(amt), "purpose": purpose, "beneficiary_bank": bank,
                                  "_gid": gid})

    # ------------------------------------------------------- 5. wallet & pipeline
    def build_wallet(self):
        for g in self.groups:
            rev = max(f["revenue_usd_m"] for f in self.financials if f["group_id"] == g["group_id"])
            for fam, k in (("GB", .0016), ("TB", .0011), ("Markets", .0008)):
                wallet = rev * 1e6 * k * self.r.uniform(.6, 1.5)
                share = self.r.uniform(.04, .35)
                for y in (2024, 2025):
                    w = wallet * (1 if y == 2025 else self.r.uniform(.85, 1.05))
                    s = share * (1 if y == 2025 else self.r.uniform(.8, 1.3))
                    self.wallet.append({"group_id": g["group_id"], "family": fam, "year": y,
                                        "wallet_usd": int(w), "our_revenue_usd": int(w * min(s, .6))})

    def build_pipeline(self):
        stages = ["Identified", "Qualified", "Proposal", "Mandated"]
        for g in self.groups:
            lead = next(c["banker_id"] for c in self.coverage if c["group_id"] == g["group_id"] and c["role"] == "Lead RM")
            in_pipe = []
            for code, _sig in self.latent.get(g["group_id"], []):
                if self.r.random() < 0.3:
                    in_pipe.append(code)
            # pipeline also contains generic RM ideas
            if self.r.random() < 0.6:
                in_pipe.append(self.r.choice(["RCF", "TRADE_FIN", "DCM", "CASH_MGMT", "VIRTUAL_ACCT"]))
            for code in dict.fromkeys(in_pipe):
                self._opp_ctr += 1
                created = self._rand_date(date(2026, 3, 1), date(2026, 10, 1))
                upd = self._rand_date(created, TODAY - timedelta(days=1))
                self.pipeline.append({"opp_id": f"OP{self._opp_ctr:03d}", "group_id": g["group_id"],
                                      "product_code": code, "stage": self.r.choice(stages),
                                      "est_revenue_usd": self.r.choice([2e5, 5e5, 1e6, 2.5e6]),
                                      "owner_banker_id": lead, "created": created.isoformat(),
                                      "updated": upd.isoformat(), "status": "Open"})

    # ------------------------------------------------------- 6. interactions & call reports
    INTENT_PHRASES = {
        "DCM": ["The CFO indicated the group is evaluating refinancing of its bonds maturing next year and asked "
                "for our view on a new USD bond issue.",
                "Treasury is preparing a refinancing plan; they want to tap the debt capital markets in H1."],
        "MNA": ["Management is actively screening acquisition targets in Southeast Asia and would welcome "
                "M&A ideas.", "The CEO hinted at a bolt-on acquisition to consolidate the regional market."],
        "ECM": ["The board is considering a listing of the regional business; they asked about IPO readiness."],
        "TRADE_FIN": ["Import volumes are growing fast and they need additional letter of credit lines.",
                      "Treasurer flagged rising trade volumes and is looking for trade finance support."],
        "COMMODITY_HEDGE": ["They are worried about input price volatility and want to hedge commodity exposure."],
        "IRS_HEDGE": ["With rates uncertain, the CFO is keen to fix the interest rate on floating-rate debt."],
        "FX_HEDGE": ["Treasurer mentioned FX volatility is hurting margins and asked about hedging programmes."],
        "SCF": ["Procurement wants to extend payment terms with suppliers without hurting the supply base - "
                "interested in supplier financing."],
        "LIQUIDITY": ["Cash is trapped across several countries; they asked about cash pooling options."],
        "CASH_MGMT": ["Subsidiaries still use several local banks; the Treasurer wants to consolidate operating "
                      "accounts."],
    }
    DECLINE_PHRASES = {
        "FX_HEDGE": "Treasurer stated the group is not interested in FX hedging at this time - board policy is to "
                    "remain unhedged.",
        "DCM": "CFO confirmed there is no plan to issue bonds; the group is deleveraging.",
        "SCF": "Procurement declined supplier financing; suppliers are paid on cash terms by policy.",
    }
    SMALLTALK = [
        "Discussed H1 trading performance; revenue broadly in line with budget.",
        "Reviewed service levels on payments; no major issues raised.",
        "Client appreciated the recent market update session.",
        "General relationship review; management changes in the finance team were discussed.",
        "Walked through our latest digital banking features.",
        "Client shared their sustainability targets for 2030.",
    ]

    def build_interactions(self):
        bk_name = {b[0]: b[1] for b in BANKERS}
        for g in self.groups:
            gid = g["group_id"]
            contacts = self.contacts_of(gid)
            team = [c["banker_id"] for c in self.coverage if c["group_id"] == gid]
            # Some contacts are deliberately neglected (no touchpoint in > 180 days)
            neglected = {c["contact_id"] for c in self.r.sample(contacts, k=max(1, len(contacts) // 3))
                         if c["title"] != "Group CFO"}
            for c in contacts:
                n = self.r.randint(2, 7) if c["title"] != "Group CFO" else self.r.randint(5, 10)
                for _ in range(n):
                    end = TODAY - timedelta(days=200) if c["contact_id"] in neglected else TODAY - timedelta(days=2)
                    d = self._rand_date(date(2024, 6, 1), end)
                    bk = self.r.choice(team) if self.r.random() < .7 else team[0]
                    self._int_ctr += 1
                    self.interactions.append({"interaction_id": f"I{self._int_ctr:05d}", "date": d.isoformat(),
                                              "channel": self.r.choice(["Meeting", "Call", "Email", "Meeting"]),
                                              "banker_id": bk, "contact_id": c["contact_id"], "group_id": gid})
            # call reports: built from a subset of meetings
            meetings = sorted([i for i in self.interactions if i["group_id"] == gid and i["channel"] == "Meeting"],
                              key=lambda x: x["date"])
            intents = [n for n in self.latent.get(gid, [])]
            decline = None
            if gid in self.latent and self.r.random() < 0.2:
                held_or_needed = {n[0] for n in intents}
                options = [p for p in self.DECLINE_PHRASES if p not in held_or_needed]
                if options:
                    decline = self.r.choice(options)
            recent_meetings = [m for m in meetings if m["date"] >= (TODAY - timedelta(days=150)).isoformat()]
            chosen = (meetings[-6:-2] if len(meetings) > 6 else meetings[:2]) + recent_meetings[-2:]
            seen = set()
            for idx, m in enumerate(chosen):
                if m["interaction_id"] in seen:
                    continue
                seen.add(m["interaction_id"])
                contact = next(c for c in contacts if c["contact_id"] == m["contact_id"])
                others = [c for c in contacts if c["contact_id"] != contact["contact_id"]]
                attendees = [contact] + (self.r.sample(others, 1) if others and self.r.random() < .5 else [])
                body = [f"Attendees: {', '.join(a['name'] + ' (' + a['title'] + ')' for a in attendees)}; "
                        f"{bk_name[m['banker_id']]} for the bank."]
                body += self.r.sample(self.SMALLTALK, 2)
                is_recent = m in recent_meetings
                if is_recent:
                    for code, sig in intents:
                        if sig == "CALL_REPORT_INTENT" or (self.r.random() < 0.25 and code in self.INTENT_PHRASES):
                            body.append(self.r.choice(self.INTENT_PHRASES[code]))
                    if decline and idx == len(chosen) - 1:
                        body.append(self.DECLINE_PHRASES[decline])
                    if gid in getattr(self, "attrition", {}):
                        body.append("Client mentioned a competitor offered sharper pricing on payments and FX.")
                for ev in self.events:
                    if ev["group_id"] == gid and 0 <= (date.fromisoformat(m["date"]) -
                                                       date.fromisoformat(ev["date"])).days < 60:
                        body.append(f"Discussed recent news: {ev['headline']}.")
                body.append("Next steps: " + self.r.choice(["circulate proposal", "set up follow-up with product "
                                                                                     "specialists",
                                                            "send market update", "no action required"]) + ".")
                self._call_ctr += 1
                self.call_reports.append({
                    "doc_id": f"CR{self._call_ctr:04d}", "doc_type": "CallReport", "date": m["date"],
                    "group_id": gid, "banker_id": m["banker_id"],
                    "attendee_contact_ids": [a["contact_id"] for a in attendees],
                    "title": f"Call report - {g['name']} - {m['date']}",
                    "text": " ".join(body)})
            if decline:
                self.gt_nba.append({"group_id": gid, "product_code": decline, "driver": "DECLINED", "realised": 0})

    def build_credit_papers(self):
        for g in self.groups:
            gid = g["group_id"]
            fin = sorted([f for f in self.financials if f["group_id"] == gid], key=lambda x: x["year"])[-1]
            ents = self.ents_of(gid)
            fac = [h for h in self.holdings if h["entity_id"] in {e["entity_id"] for e in ents}
                   and PRODUCT_BY_CODE[h["product_code"]]["family"] == "GB"]
            lev = fin["net_debt_usd_m"] / fin["ebitda_usd_m"]
            fac_txt = "; ".join(f"{PRODUCT_BY_CODE[h['product_code']]['name']} USD {h['limit_usd']/1e6:,.0f}m "
                                f"to {next(e['name'] for e in ents if e['entity_id'] == h['entity_id'])}"
                                for h in fac) or "None"
            risks = self.r.sample(["customer concentration", "FX translation risk", "refinancing risk in 2027",
                                   "commodity input costs", "regulatory change", "execution risk on expansion",
                                   "key-person dependency"], 2)
            text = (f"BORROWER: {g['name']} ({g['industry']}), headquartered in {COUNTRIES[g['hq_country']]['name']}. "
                    f"Group comprises {len(ents)} legal entities including {ents[0]['name']}"
                    + (f" and {ents[-1]['name']}" if len(ents) > 1 else "") + ". "
                    f"FINANCIALS FY{fin['year']}: revenue USD {fin['revenue_usd_m']:,.0f}m, EBITDA USD "
                    f"{fin['ebitda_usd_m']:,.0f}m, net debt USD {fin['net_debt_usd_m']:,.0f}m, leverage {lev:.1f}x. "
                    f"FACILITIES: {fac_txt}. RATING: internal rating {g['risk_rating']}. "
                    f"KEY RISKS: {risks[0]}; {risks[1]}. "
                    f"RECOMMENDATION: {'Approve renewal' if lev < 4 else 'Approve renewal with tightened covenants'} "
                    f"with leverage covenant at {max(3.5, round(lev + 1, 1))}x.")
            self.credit_papers.append({"doc_id": f"CP{gid[1:]}", "doc_type": "CreditPaper",
                                       "date": f"{fin['year'] + 1}-0{self.r.randint(3, 9)}-15",
                                       "group_id": gid, "banker_id": next(c["banker_id"] for c in self.coverage
                                                                         if c["group_id"] == gid),
                                       "attendee_contact_ids": [],
                                       "title": f"Annual credit review - {g['name']}", "text": text})

    # ------------------------------------------------------- 7. relationship intelligence plants (Experiment 3)
    def plant_relationships(self):
        """For each complex group plant 1-3 hidden connection paths to a 'Company B'."""
        for g in self.groups[:N_COMPLEX]:
            gid = g["group_id"]
            ents = self.ents_of(gid)
            b = self.r.choice([x for x in self.externals if x["kind"] in ("supplier", "target", "sponsor")])
            kinds = self.r.sample(["SUPPLIER", "SHARED_DIRECTOR", "FORMER_EMPLOYEE", "JV"], self.r.randint(1, 3))
            gold = []
            for k in kinds:
                if k == "SUPPLIER":
                    payer = self.r.choice(ents)
                    self._flow(payer, b["name"], b["country"], "USD", self.r.uniform(4e6, 2e7), "supplier", "EXTERNAL")
                    gold.append({"type": "SUPPLIER", "via": payer["name"]})
                elif k == "SHARED_DIRECTOR":
                    c = self.r.choice([c for c in self.contacts_of(gid) if c["title"] in ("CEO", "Group CFO")])
                    c["board_seats"] = ";".join(filter(None, [c["board_seats"], b["name"]]))
                    gold.append({"type": "SHARED_DIRECTOR", "via": c["name"]})
                elif k == "FORMER_EMPLOYEE":
                    c = self.r.choice([c for c in self.contacts_of(gid) if c["title"] not in ("Group CFO",)
                                       and not c["prior_employer_name"]] or self.contacts_of(gid))
                    c["prior_employer_name"] = b["name"]
                    c["prior_title"] = "Finance Manager"
                    c["prior_until"] = "2022-12-31"
                    gold.append({"type": "FORMER_EMPLOYEE", "via": c["name"]})
                elif k == "JV":
                    e = self.r.choice(ents)
                    jv = f"{g['short'].split()[0]}-{b['name'].split()[0]} JV"
                    self.jvs.append({"party_a": e["name"], "party_b": b["name"], "jv_name": jv})
                    gold.append({"type": "JV", "via": jv})
            self.gt_rel.append({"group_id": gid, "group_name": g["name"], "company_b": b["name"],
                                "connections": gold})

    # ------------------------------------------------------- 8. NBA ground truth
    def build_nba_truth(self):
        for g in self.groups[:N_NBA]:
            gid = g["group_id"]
            for code, sig in self.latent[gid]:
                if self.group_holds(gid, code):
                    continue
                p = {"UNHEDGED_FX_FLOWS": .75, "SUPPLIER_CONCENTRATION": .7, "MULTI_COUNTRY_NO_POOLING": .65,
                     "SUBSIDIARY_BANKS_ELSEWHERE": .65, "CALL_REPORT_INTENT": .75}[sig]
                self.gt_nba.append({"group_id": gid, "product_code": code, "driver": sig,
                                    "realised": int(self.r.random() < p)})
            # needs no data source hints at (irreducible noise)
            if self.r.random() < 0.25:
                code = self.r.choice(["VIRTUAL_ACCT", "CUSTODY", "RCF", "TRADE_FIN"])
                if not self.group_holds(gid, code):
                    self.gt_nba.append({"group_id": gid, "product_code": code, "driver": "UNOBSERVED",
                                        "realised": 1})

    # ------------------------------------------------------- write
    def run(self):
        self.build_groups()
        self.build_entities()
        self.build_people()
        self.build_financials_and_holdings()
        self.build_events()
        self.plan_latent_needs()
        self.build_background_deals()
        self.build_payments()
        self.plant_relationships()
        self.build_wallet()
        self.build_pipeline()
        self.build_interactions()
        self.build_credit_papers()
        self.build_nba_truth()
        return self

    def write(self):
        if SOURCE_DIR.exists():
            shutil.rmtree(SOURCE_DIR)
        if GT_DIR.exists():
            shutil.rmtree(GT_DIR)
        (SOURCE_DIR / "documents").mkdir(parents=True)
        GT_DIR.mkdir(parents=True)

        def w(path, rows, fields=None):
            fields = fields or [k for k in rows[0].keys() if not k.startswith("_")]
            with open(path, "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                wr.writeheader()
                wr.writerows(rows)

        s = SOURCE_DIR
        w(s / "crm_client_groups.csv", [{k: v for k, v in g.items() if k != "short"} for g in self.groups])
        w(s / "crm_legal_entities.csv", self.entities)
        w(s / "crm_contacts.csv", self.contacts)
        w(s / "crm_bankers.csv", [dict(zip(["banker_id", "name", "role", "business"], b)) for b in BANKERS])
        w(s / "crm_coverage.csv", self.coverage)
        w(s / "crm_pipeline.csv", self.pipeline)
        w(s / "crm_interactions.csv", self.interactions)
        w(s / "ref_external_companies.csv", self.externals)
        w(s / "ref_joint_ventures.csv", self.jvs, ["party_a", "party_b", "jv_name"])
        w(s / "payments_monthly_flows.csv", self.payments)
        w(s / "product_holdings.csv", self.holdings)
        w(s / "wallet_estimates.csv", self.wallet)
        w(s / "finance_financials.csv", self.financials)
        w(s / "deals_history.csv", self.deals)
        w(s / "news_corporate_events.csv", self.events)
        with open(s / "documents" / "call_reports.jsonl", "w") as f:
            for d in self.call_reports:
                f.write(json.dumps(d) + "\n")
        with open(s / "documents" / "credit_papers.jsonl", "w") as f:
            for d in self.credit_papers:
                f.write(json.dumps(d) + "\n")
        w(GT_DIR / "nba_realised_next12m.csv", self.gt_nba, ["group_id", "product_code", "driver", "realised"])
        w(GT_DIR / "event_outcomes.csv", self.gt_events, ["event_id", "product_code"])
        w(GT_DIR / "payment_alias_truth.csv", self.gt_aliases, ["raw_name", "entity_id"])
        with open(GT_DIR / "relationship_plants.json", "w") as f:
            json.dump(self.gt_rel, f, indent=2)
        return {
            "client_groups": len(self.groups), "legal_entities": len(self.entities),
            "external_companies": len(self.externals), "contacts": len(self.contacts),
            "product_holdings": len(self.holdings), "payment_rows": len(self.payments),
            "wallet_rows": len(self.wallet), "pipeline_opps": len(self.pipeline), "deals": len(self.deals),
            "corporate_events": len(self.events), "interactions": len(self.interactions),
            "call_reports": len(self.call_reports), "credit_papers": len(self.credit_papers),
        }


def main():
    stats = Gen().run().write()
    print("Synthetic source systems written to", SOURCE_DIR)
    for k, v in stats.items():
        print(f"  {k:20s} {v}")


if __name__ == "__main__":
    main()
