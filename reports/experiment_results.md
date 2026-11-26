# Lighthouse experiment results (synthetic data)

> All numbers below come from **synthetic data** whose hidden ground truth was generated with the same kinds of signals the graph looks for. They show that the pipeline works end-to-end and how the comparison will be run on real data - they are **not** evidence for the hypothesis itself.

## 0. Knowledge-graph build quality

- Nodes 1,866, edges 5,641, ontology violations 0
- Inferred edges: {'ULTIMATE_PARENT': 109, 'SUPPLIER_OF': 119, 'TRADES_WITH': 8}
- Intents extracted from call reports: 59
- Payment beneficiary entity resolution: precision 1.0, recall 0.972 (1404 free-text aliases)

## 1. Cross-sell / Next Best Action (25 clients, top-3)

| Method | Precision@k | Recall | F1@k | Hit-rate (clients w/ outcome) | Net-new hits (not in pipeline) | Recommended a declined product |
|---|---|---|---|---|---|---|
| Graph NBA (all sources connected) | 0.347 | 0.812 | 0.486 | 0.842 | 18 | 0 |
| Today: pipeline + product-gap | 0.24 | 0.562 | 0.336 | 0.737 | 10 | 1 |
| Ablation: payments_only | 0.2 | 0.469 | 0.28 | 0.579 | 11 | 0 |
| Ablation: call_reports_only | 0.213 | 0.5 | 0.299 | 0.632 | 11 | 0 |
| Ablation: holdings+wallet+pipeline_only | 0.227 | 0.531 | 0.318 | 0.737 | 9 | 1 |

Average distinct data sources behind each graph recommendation: 1.64

## 2. Event -> Opportunity (42 historical events)

| Method | Precision | Recall | F1 |
|---|---|---|---|
| Graph (playbook x client context as of event date) | 0.805 | 0.897 | 0.848 |
| Today: news alert -> generic product list | 0.587 | 0.91 | 0.714 |

| Event type | n | Graph P / R | Baseline P / R |
|---|---|---|---|
| ACQUISITION | 9 | 0.846 / 0.957 | 0.611 / 0.957 |
| BOND_MATURITY | 6 | 0.818 / 1.0 | 0.5 / 1.0 |
| CAPEX_PROJECT | 5 | 1.0 / 0.727 | 0.6 / 0.818 |
| DIVESTMENT | 2 | 1.0 / 1.0 | 1.0 / 1.0 |
| IPO_PLAN | 2 | 0.333 / 0.667 | 0.333 / 0.667 |
| MARKET_EXPANSION | 7 | 0.733 / 1.0 | 0.524 / 1.0 |
| MGMT_CHANGE | 2 | n/a (relationship / risk action, no product) | n/a |
| NEW_SUBSIDIARY | 3 | 0.833 / 0.833 | 0.833 / 0.833 |
| RATING_DOWNGRADE | 1 | 0.0 / 0.0 | 0.0 / 0.0 |
| SUPPLY_CHAIN_SHIFT | 5 | 0.818 / 0.9 | 0.6 / 0.9 |

Recommendations carrying a context reason: 87/87
Banker acceptance: {"status": "no feedback captured yet - use the thumbs up/down buttons in the web UI during the pilot"}

Examples of context filtering:
- `EV001` Aurora Logistics Group agrees to acquire Corvid Analytics (United Kingdom) for USD 1,200m - graph ['CASH_MGMT', 'FX_HEDGE', 'MNA'], baseline ['ACQ_FIN', 'CASH_MGMT', 'FX_HEDGE', 'MNA'], actual ['FX_HEDGE', 'MNA']; rejected: ACQ_FIN: leverage 5.0x vs 3.5x threshold (credit paper / financials)
- `EV002` Harbourline Retail Group announces entry into Mexico with new distribution hub - graph ['FX_HEDGE', 'TRADE_FIN'], baseline ['CASH_MGMT', 'FX_HEDGE', 'TRADE_FIN'], actual ['FX_HEDGE', 'TRADE_FIN']; rejected: CASH_MGMT: already bank Harbourline Retail de Mexico SA de CV in Mexico
- `EV004` Vantagepoint Healthcare Group's USD 500m notes fall due in 2026; refinancing under review - graph ['DCM', 'IRS_HEDGE'], baseline ['DCM', 'IRS_HEDGE', 'TERM_LOAN'], actual ['DCM', 'IRS_HEDGE']; rejected: TERM_LOAN: internal rating A-
- `EV005` Cedarvale Properties Group's USD 300m notes fall due in 2027; refinancing under review - graph ['DCM'], baseline ['DCM', 'IRS_HEDGE', 'TERM_LOAN'], actual ['DCM']; rejected: TERM_LOAN: internal rating BBB+; IRS_HEDGE: IRS hedge already held

## 3. Relationship Intelligence (15 complex groups, 75 multi-hop questions)

| Question template | Graph agent P / R / exact | Doc-RAG baseline P / R / exact |
|---|---|---|
| T1_entities_with_product | 1.0 / 1.0 / 15/15 | 0.788 / 0.388 / 0/15 |
| T2_connection | 0.92 / 1.0 / 13/15 | 0.0 / 0.0 / 0/15 |
| T3_contacts_not_met | 1.0 / 1.0 / 15/15 | 0.0 / 0.0 / 0/15 |
| T4_strongest_banker_cfo | 1.0 / 0.882 / 14/15 | 0.733 / 0.647 / 10/15 |
| T5_deals_for_jv_and_suppliers | 1.0 / 1.0 / 15/15 | 0.083 / 0.053 / 1/15 |
| ALL | 0.99 / 0.99 / 72/75 | 0.37 / 0.192 / 11/75 |
