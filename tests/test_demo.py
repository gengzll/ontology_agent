import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from ontology_agent.agent import offline
from ontology_agent.agent.router import route
from ontology_agent.engines import analytics as A
from ontology_agent.kg.build import extract_intents, load_graph, normalise
from ontology_agent.ontology import validate_edge


class TestOntologyAndGraph(unittest.TestCase):
    def test_graph_has_all_brief_data_types_and_no_violations(self):
        g, rep = load_graph()
        st = rep["graph"]
        self.assertEqual(st["ontology_violations"], 0)
        for cls in ["ClientGroup", "LegalEntity", "Contact", "Banker", "Product", "WalletEstimate", "Opportunity",
                    "Deal", "CorporateEvent", "CallReport", "CreditPaper", "Interaction"]:
            self.assertGreater(st["nodes_by_class"].get(cls, 0), 0, cls)
        for rel in ["PAYS", "HOLDS", "SUBSIDIARY_OF", "HAS_WALLET", "HAS_OPPORTUNITY", "EXPRESSES", "SUPPLIER_OF",
                    "ULTIMATE_PARENT"]:
            self.assertGreater(st["edges_by_relation"].get(rel, 0), 0, rel)

    def test_domain_range_validation(self):
        self.assertTrue(validate_edge("HOLDS", "LegalEntity", "Product"))
        self.assertFalse(validate_edge("HOLDS", "Contact", "Product"))
        self.assertTrue(validate_edge("MENTIONS", "CallReport", "ExternalCompany"))  # subclass of range

    def test_entity_resolution_normalisation(self):
        self.assertEqual(normalise("Aurora Logistics (Singapore) Pte Ltd"), normalise("AURORA LOGISTICS (S) PTE LTD"))

    def test_intent_extraction_handles_negation(self):
        it = extract_intents("Treasurer stated the group is not interested in FX hedging at this time.")
        self.assertEqual(it[0]["product"], "FX_HEDGE")
        self.assertEqual(it[0]["polarity"], "negative")


class TestRouter(unittest.TestCase):
    CASES = {
        "Find the latest call report for Aurora Logistics": "DOC_SEARCH",
        "Summarise the credit paper for Banyan Foods": "DOC_SUMMARY",
        "What's Cobalt Semicon's revenue?": "METRIC",
        "What products does Evergreen Textiles have?": "HOLDINGS",
        "What opportunity is implied by Aurora Logistics' subsidiaries, flows, wallet and recent events?":
            "OPPORTUNITY",
        "How are Aurora Logistics and Harrowgate Steel Trading connected?": "CONNECTION",
        "What similar deals have we executed for related clients of Banyan Foods?": "SIMILAR_DEALS",
        "What changed across my portfolio and what should I do?": "PORTFOLIO",
        "Which contacts at Jadeport Maritime have we not met in the last 6 months?": "RELATIONSHIP",
        "What opportunities does EV012 imply?": "EVENT",
        "Aurora Logistics 的收入是多少？": "METRIC",
        "Aurora Logistics 和 Harrowgate Steel Trading 有什么关联？": "CONNECTION",
    }

    def test_routes(self):
        for q, exp in self.CASES.items():
            self.assertEqual(route(q).route, exp, q)

    def test_every_route_answers_offline(self):
        for q in self.CASES:
            res = offline.answer(q)
            self.assertTrue(res["answer"].strip(), q)
            self.assertNotIn("tool `", res["answer"], q)
            self.assertTrue(res["steps"], q)


class TestAnalytics(unittest.TestCase):
    def test_event_playbook_is_time_aware(self):
        """Products won *because of* an event must not count as already-held when the event is evaluated."""
        from datetime import date
        g, _ = load_graph()
        checked = 0
        for eid in g.of_type("CorporateEvent"):
            ev = A.event_record(g, eid)
            at = date.fromisoformat(ev["date"])
            for h in A.holdings(g, ev["group_id"], as_of=at):
                self.assertLessEqual(h["since"], ev["date"])
            later = [d for d in g.ins(eid, "ORIGINATED_FROM")]
            checked += len(later)
        self.assertGreater(checked, 0)

    def test_connections_find_planted_jv(self):
        g, _ = load_graph()
        a = next(n for n in g.of_type("ClientGroup") if g.name(n).startswith("Aurora"))
        jv = [e for ent in A.entities(g, a) for e in g.outs(ent, "JV_PARTNER_WITH")]
        if not jv:
            self.skipTest("no JV planted for this seed")
        res = A.find_connections(g, a, jv[0].dst)
        self.assertIn("JV", {c["kind"] for c in res["connections"]})

    def test_nba_never_recommends_held_products(self):
        g, _ = load_graph()
        for gid in g.of_type("ClientGroup"):
            held = A.held_codes(g, gid)
            for r in A.next_best_actions(g, gid, 5):
                self.assertNotIn(r["product_code"], held)


class _FakeAnthropic(BaseHTTPRequestHandler):
    """Two-turn fake Messages API: first asks for a tool, then answers."""
    requests: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append(body)
        if len(type(self).requests) == 1:
            content = [{"type": "tool_use", "id": "toolu_1", "name": "query_metric",
                        "input": {"client": "Cobalt Semicon", "metric": "revenue"}}]
            stop = "tool_use"
        else:
            content = [{"type": "text", "text": "Revenue FY2025 is in the tool result [G003]."}]
            stop = "end_turn"
        msg = {"id": "msg_x", "type": "message", "role": "assistant", "model": body["model"], "content": content,
               "stop_reason": stop, "stop_sequence": None,
               "usage": {"input_tokens": 10, "output_tokens": 10}}
        data = json.dumps(msg).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class TestClaudeLoop(unittest.TestCase):
    def test_tool_loop_against_fake_api(self):
        try:
            import anthropic
        except ImportError:
            self.skipTest("anthropic not installed")
        from ontology_agent.agent import llm
        srv = HTTPServer(("127.0.0.1", 0), _FakeAnthropic)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        client = anthropic.Anthropic(api_key="test", base_url=f"http://127.0.0.1:{srv.server_port}", max_retries=0)
        res = llm.answer("What's Cobalt Semicon's revenue?", client=client)
        srv.shutdown()
        srv.server_close()
        client.close()
        reqs = _FakeAnthropic.requests
        self.assertEqual(len(reqs), 2)
        self.assertEqual(reqs[0]["model"], "claude-opus-5-5")
        self.assertEqual(reqs[0]["thinking"], {"type": "adaptive"})
        self.assertEqual(reqs[0]["fallbacks"], "default")
        tr = reqs[1]["messages"][-1]["content"][0]
        self.assertEqual(tr["type"], "tool_result")
        self.assertIn("revenue", tr["content"])
        self.assertEqual(res["steps"][0]["tool"], "query_metric")
        self.assertIn("[G003]", res["answer"])


if __name__ == "__main__":
    unittest.main()
