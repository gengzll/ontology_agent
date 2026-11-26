"""A small in-memory property graph with ontology validation.

Stand-in for a production graph database (Neo4j / Neptune / Stardog / TigerGraph).
Kept dependency-free so the demo runs anywhere.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

from ..ontology import CLASSES, RELATIONS, validate_edge


@dataclass
class Edge:
    src: str
    rel: str
    dst: str
    props: dict = field(default_factory=dict)


class KnowledgeGraph:
    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.out: dict[str, list[Edge]] = defaultdict(list)
        self.inc: dict[str, list[Edge]] = defaultdict(list)
        self.by_type: dict[str, list[str]] = defaultdict(list)
        self.violations: list[str] = []

    # ---------------------------------------------------------------- write
    def add_node(self, nid: str, ntype: str, **props) -> str:
        if ntype not in CLASSES:
            raise ValueError(f"Unknown ontology class {ntype}")
        if nid in self.nodes:
            self.nodes[nid]["props"].update({k: v for k, v in props.items() if v not in (None, "")})
        else:
            self.nodes[nid] = {"id": nid, "type": ntype, "props": props}
            self.by_type[ntype].append(nid)
        return nid

    def add_edge(self, src: str, rel: str, dst: str, **props) -> Edge | None:
        if src not in self.nodes or dst not in self.nodes:
            self.violations.append(f"dangling {src}-{rel}->{dst}")
            return None
        if not validate_edge(rel, self.nodes[src]["type"], self.nodes[dst]["type"]):
            self.violations.append(f"ontology violation {self.nodes[src]['type']}-{rel}->{self.nodes[dst]['type']}")
            return None
        e = Edge(src, rel, dst, props)
        self.out[src].append(e)
        self.inc[dst].append(e)
        return e

    # ---------------------------------------------------------------- read
    def node(self, nid: str) -> dict:
        return self.nodes[nid]

    def name(self, nid: str) -> str:
        n = self.nodes.get(nid)
        if not n:
            return nid
        p = n["props"]
        return p.get("name") or p.get("title") or p.get("headline") or p.get("code") or nid

    def type(self, nid: str) -> str:
        return self.nodes[nid]["type"]

    def outs(self, nid: str, rel: str | None = None) -> list[Edge]:
        return [e for e in self.out.get(nid, []) if rel is None or e.rel == rel]

    def ins(self, nid: str, rel: str | None = None) -> list[Edge]:
        return [e for e in self.inc.get(nid, []) if rel is None or e.rel == rel]

    def neighbours(self, nid: str, rels: set[str] | None = None) -> list[tuple[Edge, str, str]]:
        """Return (edge, neighbour_id, direction) for both directions."""
        res = [(e, e.dst, "out") for e in self.out.get(nid, []) if rels is None or e.rel in rels]
        res += [(e, e.src, "in") for e in self.inc.get(nid, []) if rels is None or e.rel in rels]
        return res

    def of_type(self, ntype: str) -> list[str]:
        return list(self.by_type.get(ntype, []))

    def find_paths(self, a: str, b: str, max_hops: int = 4, max_paths: int = 8,
                   exclude_rels: set[str] | None = None, avoid_types: set[str] | None = None) -> list[list]:
        """All simple shortest-ish paths (BFS, up to max_hops) between two nodes.

        A path is a list alternating [node, (rel, direction), node, ...].
        Hub node types (Country, Industry, Currency, Product) are avoided as intermediate nodes by default so
        that 'both are in Singapore' is not reported as a relationship.
        """
        exclude_rels = exclude_rels or set()
        avoid_types = avoid_types if avoid_types is not None else {"Country", "Industry", "Currency", "Product",
                                                                   "Intent", "WalletEstimate"}
        paths: list[list] = []
        q = deque([[a]])
        best = None
        while q and len(paths) < max_paths:
            path = q.popleft()
            hops = (len(path) - 1) // 2
            if best is not None and hops > best + 1:
                break
            last = path[-1]
            if hops >= max_hops:
                continue
            for e, nb, d in self.neighbours(last):
                if e.rel in exclude_rels or nb in path[::2]:
                    continue
                if nb != b and self.nodes[nb]["type"] in avoid_types:
                    continue
                new = path + [(e.rel, d, e.props), nb]
                if nb == b:
                    paths.append(new)
                    best = hops + 1 if best is None else best
                else:
                    q.append(new)
        return paths

    def stats(self) -> dict:
        rel_counts: dict[str, int] = defaultdict(int)
        for es in self.out.values():
            for e in es:
                rel_counts[e.rel] += 1
        return {"nodes": len(self.nodes), "edges": sum(rel_counts.values()),
                "nodes_by_class": {k: len(v) for k, v in sorted(self.by_type.items())},
                "edges_by_relation": dict(sorted(rel_counts.items())),
                "ontology_violations": len(self.violations)}

    def subgraph(self, center: str, depth: int = 1, limit: int = 120,
                 rels: set[str] | None = None) -> dict:
        """Neighbourhood for visualisation."""
        seen = {center}
        frontier = [center]
        edges = []
        for _ in range(depth):
            nxt = []
            for n in frontier:
                for e, nb, _d in self.neighbours(n, rels):
                    if len(seen) >= limit and nb not in seen:
                        continue
                    edges.append(e)
                    if nb not in seen:
                        seen.add(nb)
                        nxt.append(nb)
            frontier = nxt
        uniq = {(e.src, e.rel, e.dst): e for e in edges if e.src in seen and e.dst in seen}
        return {
            "nodes": [{"id": n, "label": self.name(n), "type": self.type(n)} for n in seen],
            "edges": [{"source": e.src, "target": e.dst, "rel": e.rel} for e in uniq.values()],
        }

    @staticmethod
    def relation_spec(rel: str) -> dict:
        return RELATIONS[rel]
