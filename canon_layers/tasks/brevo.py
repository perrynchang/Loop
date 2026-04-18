"""
Task Brevo: Mental Reasoning Breadth
Recursive traversal of directed acyclic graphs (DAGs).
Format: <bos> x1 y1 x2 y2 ... xm ym <query> q <ans> a1 a2 ... ap <eos>
Output: all vertices reachable from q, sorted in topological order (leaves first).
"""
import random
from collections import defaultdict, deque
import torch
from torch.utils.data import IterableDataset


class BrevoTokenizer:
    def __init__(self, variant="brevo1"):
        self.variant = variant
        if variant == "brevo1":
            self.vocab_size = 256  # single token per vertex
            self.node_min_len = 1
            self.node_max_len = 1
        else:  # brevo2
            self.vocab_size = 4
            self.node_min_len = 2
            self.node_max_len = 4

        self.BOS = 0
        self.EOS = 1
        self.QUERY = 2
        self.ANS = 3
        self.NODE_OFFSET = 4
        self.total_vocab = self.NODE_OFFSET + self.vocab_size + 10

    def encode_node(self, node_id, rng=None):
        if self.node_min_len == self.node_max_len == 1:
            return [self.NODE_OFFSET + (node_id % self.vocab_size)]
        length = rng.randint(self.node_min_len, self.node_max_len) if rng else self.node_min_len
        r = random.Random(node_id)
        return [self.NODE_OFFSET + r.randint(0, self.vocab_size - 1) for _ in range(length)]


def build_random_dag(n, max_degree=4, rng=None):
    """Build a random DAG on n nodes with topological order 0 < 1 < ... < n-1."""
    if rng is None:
        rng = random
    edges = []  # list of (u, v) where u < v
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)

    for v in range(1, n):
        possible = [u for u in range(v) if out_degree[u] < max_degree]
        if not possible:
            continue
        n_parents = rng.randint(1, min(max_degree, len(possible)))
        parents = rng.sample(possible, n_parents)
        for u in parents:
            edges.append((u, v))
            out_degree[u] += 1

    return edges


def topological_reachable(q, children, n):
    """Return all nodes reachable from q in topological order (leaves first)."""
    # BFS to find all reachable nodes
    reachable = set()
    stack = [q]
    while stack:
        node = stack.pop()
        if node in reachable:
            continue
        reachable.add(node)
        stack.extend(children[node])

    # Topological sort on reachable set (ancestors before q)
    # Since DAG is sorted: node < children, we reverse for leaf-first output
    sorted_nodes = sorted(reachable, reverse=True)
    # Return all except q itself
    return [v for v in sorted_nodes if v != q]


class BrevoDataset(IterableDataset):
    def __init__(self, N, variant="brevo1", context_len=1024, seed=42):
        super().__init__()
        self.N = N
        self.tokenizer = BrevoTokenizer(variant)
        self.context_len = context_len
        self.seed = seed

    def _make_instance(self, rng):
        tok = self.tokenizer
        n = rng.randint(3, self.N)
        edges = build_random_dag(n, max_degree=4, rng=rng)

        # Build children map
        children = defaultdict(list)
        for u, v in edges:
            children[u].append(v)

        # Pick a query node with non-empty reachable set
        nodes_with_children = [u for u in range(n) if children[u]]
        if not nodes_with_children:
            return None
        q = rng.choice(nodes_with_children)
        answer = topological_reachable(q, children, n)
        if not answer:
            return None

        # Encode
        tokens = [tok.BOS]
        edge_list = list(edges)
        rng.shuffle(edge_list)
        for u, v in edge_list:
            tokens += tok.encode_node(u, rng)
            tokens += tok.encode_node(v, rng)

        tokens.append(tok.QUERY)
        tokens += tok.encode_node(q, rng)
        tokens.append(tok.ANS)
        for a in answer:
            tokens += tok.encode_node(a, rng)
        tokens.append(tok.EOS)
        return tokens

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        seed = self.seed + (worker_info.id if worker_info else 0)
        rng = random.Random(seed)

        buffer = []
        while True:
            inst = self._make_instance(rng)
            if inst is None:
                continue
            buffer.extend(inst)
            while len(buffer) >= self.context_len:
                chunk = buffer[:self.context_len]
                buffer = buffer[self.context_len:]
                yield torch.tensor(chunk, dtype=torch.long)


def build_brevo_dataset(variant="brevo1", N=70, context_len=1024, seed=42):
    return BrevoDataset(N, variant, context_len, seed=seed)
