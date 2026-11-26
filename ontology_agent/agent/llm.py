"""Claude-powered ontology agent (Messages API + client tools, manual loop).

The loop is hosted here (not on Managed Agents) because every tool reads the
in-process knowledge graph. The router's suggestion is given to Claude as a hint;
Claude decides which tools to call and writes the final, evidence-cited answer.
"""

from __future__ import annotations

import json
import os
import time

from .. import AS_OF
from ..ontology import CLASSES, RELATIONS
from . import tools as T
from .router import ROUTES, route

MODEL = os.environ.get("ONTOLOGY_AGENT_MODEL", "claude-opus-5-5")
MAX_TURNS = 12
MAX_TOOL_CHARS = 15000


def system_prompt() -> str:
    classes = ", ".join(CLASSES)
    rels = "; ".join(f"{k}({'/'.join(v['domain'])}->{'/'.join(v['range'])})" for k, v in RELATIONS.items())
    table = "\n".join(f"- {k}: {v['technology']} - e.g. \"{v['example']}\"" for k, v in ROUTES.items())
    return f"""You are the Frontline Client Intelligence agent for a Corporate & Investment Bank (CIB).
You answer banker questions by querying a shared CIB client ontology + knowledge graph through tools.
All data is synthetic demo data; today's date is {AS_OF}.

Ontology classes: {classes}
Ontology relations: {rels}
Product families: GB (Global Banking), TB (Transaction Banking), Markets.

Pick the cheapest technology that can answer reliably (routing table):
{table}

Rules:
- Ground every statement in tool output. Never invent numbers, names, deals or dates.
- Cite graph node ids in square brackets, e.g. [G001], [CR0012], [EV043], [D0021].
- For opportunity questions, combine sources (ownership, payments, holdings, wallet, pipeline, call reports,
  events) and say which sources support each recommendation; mention anything suppressed (client declined,
  credit gate) and why.
- For connection questions, give each path explicitly (A -REL-> X -REL-> B) and rank direct before indirect.
- If an entity cannot be resolved, say so and suggest the closest matches.
- Answer in the language of the question. Be concise: a short answer first, then evidence bullets,
  then one 'Suggested next step' line where relevant. Use Markdown."""


def _client():
    import anthropic  # optional dependency
    return anthropic.Anthropic()


def llm_available() -> bool:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                or os.environ.get("ANTHROPIC_PROFILE"))


def answer(question: str, persona_banker: str = "BK01", client=None) -> dict:
    import anthropic

    client = client or _client()
    r = route(question)
    hint = (f"[Router hint] route={r.route} technology=\"{r.technology}\"; linked entities="
            + json.dumps([{k: v for k, v in e.items() if k != 'pos'} for e in r.entities])
            + f"; the asking banker is {T.G().name(persona_banker)} [{persona_banker}].")
    messages: list[dict] = [{"role": "user", "content": [{"type": "text", "text": question},
                                                         {"type": "text", "text": hint}]}]
    tools = T.claude_tool_defs()
    steps, highlight = [], [e["id"] for e in r.entities]
    final_text, stop = "", None
    for _turn in range(MAX_TURNS):
        try:
            resp = client.beta.messages.create(
                model=MODEL,
                max_tokens=16000,
                system=[{"type": "text", "text": system_prompt(), "cache_control": {"type": "ephemeral"}}],
                tools=tools,
                messages=messages,
                thinking={"type": "adaptive"},
                output_config={"effort": "medium"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.RateLimitError as e:
            return _error(question, r, steps, f"Rate limited by the API: {e}")
        except anthropic.APIStatusError as e:
            return _error(question, r, steps, f"API error {e.status_code}: {e.message}")
        except anthropic.APIConnectionError as e:
            return _error(question, r, steps, f"Could not reach the API: {e}")
        stop = resp.stop_reason
        if stop == "refusal":
            final_text = "The model declined this request."
            break
        messages.append({"role": "assistant", "content": resp.content})
        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        if stop != "tool_use" or not tool_uses:
            final_text = "".join(b.text for b in resp.content if b.type == "text")
            if stop == "max_tokens":
                final_text += "\n\n_(answer truncated: max_tokens reached)_"
            break
        results = []
        for tu in tool_uses:
            t0 = time.perf_counter()
            args = tu.input if isinstance(tu.input, dict) else {}
            try:
                out = T.call_tool(tu.name, args)
                content, is_err = json.dumps(out, default=str), False
                highlight += T.node_ids_in(out)[:20]
            except Exception as e:  # report tool failures back to the model
                content, is_err = f"Error: {e}", True
            if len(content) > MAX_TOOL_CHARS:
                content = content[:MAX_TOOL_CHARS] + '..."[truncated]'
            steps.append({"tool": tu.name, "args": args, "ms": round((time.perf_counter() - t0) * 1000, 1),
                          "error": content if is_err else None, "result_preview": content[:600]})
            results.append({"type": "tool_result", "tool_use_id": tu.id, "content": content, "is_error": is_err})
        messages.append({"role": "user", "content": results})  # all results in one message
    else:
        final_text = final_text or "Stopped after the maximum number of tool turns."
    return {"question": question, "route": r.route, "technology": r.technology, "router_scores": r.scores,
            "linked_entities": [{k: v for k, v in e.items() if k != "pos"} for e in r.entities],
            "mode": f"claude ({MODEL})", "steps": steps, "answer": final_text, "answer_items": [],
            "highlight": list(dict.fromkeys(highlight))[:60], "stop_reason": stop}


def _error(question, r, steps, msg):
    return {"question": question, "route": r.route, "technology": r.technology, "router_scores": r.scores,
            "linked_entities": [], "mode": "claude (error)", "steps": steps, "answer": f"**{msg}**",
            "answer_items": [], "highlight": []}
