"""Search / vector RAG over unstructured content (TF-IDF cosine, dependency-free).

This plays the role of today's "Client Advisor" document RAG: it is good at
finding and summarising documents but has no notion of how entities connect.
Swap in a real embedding model + vector DB in production.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache

from ..kg.build import load_graph

STOP = set("""a an the of to in on for and or with by at from is are was were be been this that these those it its
as we our they their them has have had not no will would can could should into than then there here about
which who whom what when where how all any each per via""".split())


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9&]+", text.lower())
    return [t for t in toks if t not in STOP and len(t) > 1]


class VectorIndex:
    def __init__(self, docs: list[dict]):
        self.docs = docs
        self.df = Counter()
        tfs = []
        for d in docs:
            tf = Counter(tokenize(d["title"] + " " + d["text"]))
            tfs.append(tf)
            self.df.update(tf.keys())
        n = len(docs)
        self.idf = {t: math.log((n + 1) / (c + 0.5)) for t, c in self.df.items()}
        self.vecs = [self._vec(tf) for tf in tfs]

    def _vec(self, tf: Counter) -> dict:
        v = {t: (1 + math.log(c)) * self.idf.get(t, 0) for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1
        return {t: x / norm for t, x in v.items()}

    def search(self, query: str, k: int = 5, group_id: str | None = None, doc_type: str | None = None,
               sort_by_date: bool = False) -> list[dict]:
        q = self._vec(Counter(tokenize(query)))
        res = []
        for d, v in zip(self.docs, self.vecs):
            if group_id and d.get("group_id") != group_id:
                continue
            if doc_type and d["doc_type"] != doc_type:
                continue
            score = sum(w * v.get(t, 0) for t, w in q.items())
            res.append((score, d))
        if sort_by_date:
            res.sort(key=lambda x: (x[1]["date"], x[0]), reverse=True)
        else:
            res.sort(key=lambda x: x[0], reverse=True)
        return [{"doc_id": d["doc_id"], "doc_type": d["doc_type"], "date": d["date"], "title": d["title"],
                 "group_id": d.get("group_id"), "score": round(s, 3), "text": d["text"]} for s, d in res[:k]]


@lru_cache(maxsize=1)
def index() -> VectorIndex:
    g, _ = load_graph()
    docs = []
    for t in ("CallReport", "CreditPaper"):
        for nid in g.of_type(t):
            p = g.node(nid)["props"]
            docs.append({"doc_id": nid, "doc_type": t, "date": p["date"], "title": p["title"], "text": p["text"],
                         "group_id": p["group_id"]})
    # deal tombstones and news are also indexed (Client Advisor can see them as text)
    for nid in g.of_type("Deal"):
        p = g.node(nid)["props"]
        ent = g.outs(nid, "EXECUTED_FOR")[0].dst
        gid = g.outs(ent, "BELONGS_TO_GROUP")[0].dst
        docs.append({"doc_id": nid, "doc_type": "DealTombstone", "date": p["date"], "title": f"Deal {nid}",
                     "text": p["summary"], "group_id": gid})
    for nid in g.of_type("CorporateEvent"):
        p = g.node(nid)["props"]
        gid = g.outs(nid, "AFFECTS")[0].dst
        docs.append({"doc_id": nid, "doc_type": "News", "date": p["date"], "title": p["headline"],
                     "text": p["headline"], "group_id": gid})
    return VectorIndex(docs)


def summarise_extractive(text: str, max_sentences: int = 5) -> list[str]:
    """Fallback summariser used when no LLM is configured: keep labelled / signal-bearing sentences."""
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    key = re.compile(r"FINANCIALS|FACILITIES|RATING|KEY RISKS|RECOMMENDATION|refinanc|acqui|hedg|listing|"
                     r"trade|cash|competitor|next steps", re.I)
    picked = [s for s in sents if key.search(s)]
    return (picked or sents)[:max_sentences]
