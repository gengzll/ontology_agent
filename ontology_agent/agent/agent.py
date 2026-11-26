"""Facade: use Claude when credentials are configured, otherwise the deterministic planner."""

from __future__ import annotations

from . import llm, offline


def ask(question: str, mode: str = "auto", persona_banker: str = "BK01") -> dict:
    """mode: 'auto' | 'offline' | 'claude'."""
    use_llm = mode == "claude" or (mode == "auto" and llm.llm_available())
    if use_llm:
        res = llm.answer(question, persona_banker)
        if res.get("mode") == "claude (error)" and mode == "auto":
            fb = offline.answer(question, persona_banker)
            fb["answer"] = res["answer"] + "\n\n_Falling back to the offline planner:_\n\n" + fb["answer"]
            return fb
        return res
    return offline.answer(question, persona_banker)
