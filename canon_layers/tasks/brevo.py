"""
Task Brevo: Mental Reasoning Breadth
Recursive traversal of directed acyclic graphs (DAGs).
Format: <bos> x1 y1 x2 y2 ... xm ym <query> q <ans> a1 a2 ... ap <eos>
Output: all ancestors of q plus q itself, sorted in topological order (roots first, q last).
"""
import random
from collections import defaultdict
import torch
from torch.utils.data import IterableDataset


class BrevoTokenizer:
    def __init__(self, variant="brevo1", N=None):
        self.variant = variant
        if variant == "brevo1":
            self.vocab_size = N if N is not None else 256  # paper: one token per vertex, vocab = N
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
        # brevo1: node tokens occupy [NODE_OFFSET, NODE_OFFSET + vocab_size)
        # brevo2: boundary tokens reach NODE_OFFSET + 2*vocab_size
        if variant == "brevo1":
            self.total_vocab = self.NODE_OFFSET + self.vocab_size
        else:
            self.total_vocab = self.NODE_OFFSET + 2 * self.vocab_size + 1

    def encode_node(self, node_id, rng=None):
        return [self.NODE_OFFSET + (node_id % self.vocab_size)]


def _generate_brevo_words(rng, n, mini_vocab, min_len, max_len, offset):
    """Generate n unique multi-token words; last token of each word has +mini_vocab boundary marker."""
    def sample_word(length):
        toks = [offset + rng.randint(1, mini_vocab) for _ in range(length)]
        toks[-1] += mini_vocab  # word-boundary marker
        return tuple(toks)

    words = set()
    while len(words) < n:
        words.add(sample_word(rng.randint(min_len, max_len)))
    return sorted(words)


def build_random_dag(n, max_degree=4, rng=None):
    """Build a random DAG on n nodes with topological order 0 < 1 < ... < n-1.

    Matches the author's 'leaves_on_left' constraint: a random number of the
    first nodes (1 to ~n/4) are guaranteed structural leaves (no parents).
    """
    if rng is None:
        rng = random
    edges = []  # list of (u, v) where u < v
    out_degree = defaultdict(int)

    # First `leaves` nodes get no parents — they are guaranteed leaves.
    leaves = rng.randint(1, (n - 1) // 4 + 1)
    for v in range(leaves, n):
        possible = [u for u in range(v) if out_degree[u] < max_degree]
        if not possible:
            continue
        n_parents = rng.randint(1, min(max_degree, len(possible)))
        parents = rng.sample(possible, n_parents)
        for u in parents:
            edges.append((u, v))
            out_degree[u] += 1

    return edges


def topological_reachable(q, parents, rng):
    """Return all ancestors of q plus q, in a random valid topological order (roots first, q last)."""
    reachable = set()
    stack = [q]
    while stack:
        node = stack.pop()
        if node in reachable:
            continue
        reachable.add(node)
        stack.extend(parents[node])

    # Build children map and in-degrees within the reachable subgraph
    children_sub = defaultdict(list)
    in_degree = {node: 0 for node in reachable}
    for node in reachable:
        for parent in parents[node]:
            if parent in reachable:
                children_sub[parent].append(node)
                in_degree[node] += 1

    # Randomized Kahn's: pop a random zero-in-degree node each step
    queue = [node for node in reachable if in_degree[node] == 0]
    order = []
    while queue:
        node = queue.pop(rng.randint(0, len(queue) - 1))
        order.append(node)
        for child in children_sub[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
    return order


class BrevoDataset(IterableDataset):
    def __init__(self, N, variant="brevo1", context_len=1024, seed=42):
        super().__init__()
        self.N = N
        self.tokenizer = BrevoTokenizer(variant, N=N)
        self.context_len = context_len
        self.seed = seed

    def _make_instance(self, rng):
        """Return (tokens, mask) for one instance. mask=1 for <ans>, answer tokens, and <eos>."""
        tok = self.tokenizer
        # Curriculum: n ∝ 1/(n + √N), matching author's distribution
        ns = list(range(3, self.N + 1))
        w = [1.0 / (n + self.N ** 0.5 + 1e-12) for n in ns]
        n = rng.choices(ns, weights=w)[0]
        edges = build_random_dag(n, max_degree=4, rng=rng)

        # Build parents map (child -> list of parents)
        parents = defaultdict(list)
        for u, v in edges:
            parents[v].append(u)

        # Pick a query node from the last ~25% of construction-order nodes
        # (matching the author: these nodes tend to have more ancestors, harder queries).
        start = max(3 * n // 4, n - 1)
        candidates = [v for v in range(start, n) if parents[v]]
        if not candidates:
            return None
        q = rng.choice(candidates)
        answer = topological_reachable(q, parents, rng)
        if not answer:
            return None

        if tok.variant == "brevo2":
            words = _generate_brevo_words(
                rng, n, tok.vocab_size, tok.node_min_len, tok.node_max_len, tok.NODE_OFFSET
            )
            rng.shuffle(words)  # randomize word→node assignment (set iteration order is not random)
            word_map = {node_id: list(words[node_id]) for node_id in range(n)}
            encode = word_map.__getitem__
        else:
            # Randomize node names: sample n distinct IDs from the vocabulary so the
            # model cannot learn shortcuts from the sequential node-index → token mapping.
            perm = rng.sample(range(tok.vocab_size), n)
            encode = lambda node_id: [tok.NODE_OFFSET + perm[node_id]]

        # Encode
        tokens = [tok.BOS]
        mask = [0]
        edge_list = list(edges)
        rng.shuffle(edge_list)
        for u, v in edge_list:
            u_toks, v_toks = encode(u), encode(v)
            tokens += u_toks + v_toks
            mask += [0] * (len(u_toks) + len(v_toks))

        tokens.append(tok.QUERY)
        mask.append(0)
        q_toks = encode(q)
        tokens += q_toks
        mask += [0] * len(q_toks)

        tokens.append(tok.ANS)
        mask.append(0)
        for a in answer:
            a_toks = encode(a)
            tokens += a_toks
            mask += [1] * len(a_toks)
        tokens.append(tok.EOS)
        mask.append(1)

        return tokens, mask

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        seed = self.seed + (worker_info.id if worker_info else 0)
        rng = random.Random(seed)

        bos = self.tokenizer.BOS
        token_buf, mask_buf = [], []
        while True:
            inst = self._make_instance(rng)
            if inst is None:
                continue
            toks, mask = inst
            token_buf.extend(toks)
            mask_buf.extend(mask)
            while len(token_buf) >= self.context_len:
                chunk_toks = token_buf[:self.context_len]
                chunk_mask = mask_buf[:self.context_len]
                rest_toks = token_buf[self.context_len:]
                rest_mask = mask_buf[self.context_len:]
                # Left-align: discard partial instance tail so next chunk starts at BOS
                next_bos = next((i for i, t in enumerate(rest_toks) if t == bos), None)
                if next_bos is None:
                    token_buf, mask_buf = [], []
                else:
                    token_buf = rest_toks[next_bos:]
                    mask_buf = rest_mask[next_bos:]
                yield (
                    torch.tensor(chunk_toks, dtype=torch.long),
                    torch.tensor(chunk_mask, dtype=torch.bool),
                )


def build_brevo_dataset(variant="brevo1", N=70, context_len=1024, seed=42):
    return BrevoDataset(N, variant, context_len, seed=seed)
