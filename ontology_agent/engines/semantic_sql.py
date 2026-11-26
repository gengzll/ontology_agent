"""SQL + semantic layer over the structured source systems (SQLite in memory).

The semantic layer maps business terms ("revenue", "share of wallet",
"payment volume") to governed SQL so that the agent never improvises metric
definitions.
"""

from __future__ import annotations

import csv
import sqlite3
from functools import lru_cache

from .. import SOURCE_DIR

TABLES = {
    "client_groups": "crm_client_groups.csv",
    "legal_entities": "crm_legal_entities.csv",
    "contacts": "crm_contacts.csv",
    "bankers": "crm_bankers.csv",
    "coverage": "crm_coverage.csv",
    "pipeline": "crm_pipeline.csv",
    "interactions": "crm_interactions.csv",
    "external_companies": "ref_external_companies.csv",
    "payments": "payments_monthly_flows.csv",
    "holdings": "product_holdings.csv",
    "wallet": "wallet_estimates.csv",
    "financials": "finance_financials.csv",
    "deals": "deals_history.csv",
    "events": "news_corporate_events.csv",
}

# Governed metric definitions: business term -> SQL (parameter :gid = group id)
METRICS: dict[str, dict] = {
    "revenue": {"desc": "Annual revenue (USD m) from audited financials",
                "sql": "SELECT year, revenue_usd_m AS value FROM financials WHERE group_id=:gid ORDER BY year",
                "unit": "USD m"},
    "ebitda": {"desc": "Annual EBITDA (USD m)",
               "sql": "SELECT year, ebitda_usd_m AS value FROM financials WHERE group_id=:gid ORDER BY year",
               "unit": "USD m"},
    "net_debt": {"desc": "Net debt (USD m)",
                 "sql": "SELECT year, net_debt_usd_m AS value FROM financials WHERE group_id=:gid ORDER BY year",
                 "unit": "USD m"},
    "leverage": {"desc": "Net debt / EBITDA (x)",
                 "sql": "SELECT year, ROUND(net_debt_usd_m*1.0/ebitda_usd_m,2) AS value FROM financials "
                        "WHERE group_id=:gid ORDER BY year", "unit": "x"},
    "wallet": {"desc": "Estimated fee wallet by product family (USD)",
               "sql": "SELECT year, family, wallet_usd AS value FROM wallet WHERE group_id=:gid ORDER BY year, family",
               "unit": "USD"},
    "share_of_wallet": {"desc": "Our revenue / estimated wallet by product family",
                        "sql": "SELECT year, family, ROUND(our_revenue_usd*1.0/wallet_usd,3) AS value FROM wallet "
                               "WHERE group_id=:gid ORDER BY year, family", "unit": "ratio"},
    "bank_revenue": {"desc": "Our revenue from the client by product family (USD)",
                     "sql": "SELECT year, family, our_revenue_usd AS value FROM wallet WHERE group_id=:gid "
                            "ORDER BY year, family", "unit": "USD"},
    "payment_volume": {"desc": "Monthly outgoing payment volume through us (USD)",
                       "sql": "SELECT p.month, SUM(p.amount_usd) AS value FROM payments p JOIN legal_entities e "
                              "ON e.entity_id=p.payer_entity_id WHERE e.group_id=:gid GROUP BY p.month "
                              "ORDER BY p.month", "unit": "USD"},
    "product_holdings": {"desc": "Products held per legal entity",
                         "sql": "SELECT e.name AS entity, e.country, h.product_code, h.since FROM holdings h JOIN "
                                "legal_entities e ON e.entity_id=h.entity_id WHERE e.group_id=:gid "
                                "ORDER BY e.name, h.product_code", "unit": ""},
}

METRIC_SYNONYMS = {
    "revenue": ["revenue", "sales", "turnover", "top line", "收入", "营收", "营业额"],
    "ebitda": ["ebitda", "利润"],
    "net_debt": ["net debt", "净负债", "debt"],
    "leverage": ["leverage", "杠杆"],
    "share_of_wallet": ["share of wallet", "wallet share", "sow", "钱包份额"],
    "wallet": ["wallet", "fee pool", "钱包"],
    "bank_revenue": ["our revenue", "revenue from", "income from", "我们的收入"],
    "payment_volume": ["payment volume", "payments", "flows", "支付", "流水"],
}


@lru_cache(maxsize=1)
def connection() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:", check_same_thread=False)
    for table, fname in TABLES.items():
        with open(SOURCE_DIR / fname, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        cols = list(rows[0].keys())

        def typed(v):
            if len(v) > 15:  # identifiers such as LEIs stay text
                return v
            try:
                return int(v)
            except ValueError:
                try:
                    return float(v)
                except ValueError:
                    return v
        con.execute(f"CREATE TABLE {table} ({', '.join(cols)})")
        con.executemany(f"INSERT INTO {table} VALUES ({', '.join('?' * len(cols))})",
                        [[typed(r[c]) for c in cols] for r in rows])
    return con


def schema() -> dict:
    con = connection()
    out = {}
    for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        out[t] = [r[1] for r in con.execute(f"PRAGMA table_info({t})")]
    return out


def run_sql(sql: str, params: dict | None = None, limit: int = 200) -> dict:
    s = sql.strip().rstrip(";")
    if not s.lower().startswith(("select", "with")) or ";" in s:
        return {"error": "Only a single read-only SELECT statement is allowed."}
    try:
        cur = connection().execute(s, params or {})
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchmany(limit)]
        return {"sql": s, "columns": cols, "rows": rows}
    except sqlite3.Error as e:
        return {"error": str(e), "sql": s}


def metric_from_text(text: str) -> str | None:
    t = text.lower()
    best = None
    for m, syns in METRIC_SYNONYMS.items():
        for s in syns:
            if s in t and (best is None or len(s) > best[1]):
                best = (m, len(s))
    return best[0] if best else None


def query_metric(metric: str, group_id: str) -> dict:
    spec = METRICS.get(metric)
    if not spec:
        return {"error": f"Unknown metric '{metric}'. Known: {sorted(METRICS)}"}
    res = run_sql(spec["sql"], {"gid": group_id})
    res.update({"metric": metric, "definition": spec["desc"], "unit": spec["unit"]})
    return res
