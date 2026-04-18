"""
Task Depo: Mental Reasoning Depth
k-hop traversal over directed permutations.
Format: <bos> x1 y1 x2 y2 ... xn yn <query_k1> q1 a1 <query_k2> q2 a2 ... <eos>
"""
import random
import torch
from torch.utils.data import Dataset, IterableDataset


class DepoTokenizer:
    """Simple tokenizer for Depo task."""

    def __init__(self, variant="depo1"):
        self.variant = variant
        if variant == "depo1":
            self.vocab_size = 50
            self.node_min_len = 1
            self.node_max_len = 2
        else:  # depo2
            self.vocab_size = 4
            self.node_min_len = 5
            self.node_max_len = 7

        # Special tokens
        self.BOS = 0
        self.EOS = 1
        self.NODE_VOCAB_OFFSET = 2  # node tokens start here
        # Query tokens: one per hop depth k=1..K_max
        self.K_MAX = 16
        self.QUERY_OFFSET = self.NODE_VOCAB_OFFSET + self.vocab_size
        # Total vocab: BOS, EOS, node_vocab, query_k tokens
        self.total_vocab = self.QUERY_OFFSET + self.K_MAX + 1

    def encode_node(self, node_id):
        """Encode a node as a sequence of tokens."""
        length = random.randint(self.node_min_len, self.node_max_len)
        # Use node_id as seed so each node always encodes the same way in one instance
        rng = random.Random(node_id)
        return [self.NODE_VOCAB_OFFSET + rng.randint(0, self.vocab_size - 1) for _ in range(length)]

    def query_token(self, k):
        return self.QUERY_OFFSET + k


class DepoDataset(IterableDataset):
    """
    Generates Depo task instances on-the-fly.

    Each instance:
      <bos> x1_toks y1_toks x2_toks y2_toks ... xn_toks yn_toks
      <query_k1> q1_toks a1_toks <query_k2> q2_toks a2_toks ... <eos>

    The permutation is a bijection perm: [0..n-1] -> [0..n-1].
    Edge xi->yi means node xi maps to yi (1-hop successor).
    The k-th successor of a query node q is the answer.
    """

    def __init__(self, N, K, variant="depo1", context_len=2048,
                 n_queries_per_instance=4, seed=42):
        super().__init__()
        self.N = N
        self.K = K
        self.tokenizer = DepoTokenizer(variant)
        self.context_len = context_len
        self.n_queries = n_queries_per_instance
        self.seed = seed

    def _make_instance(self, rng):
        tok = self.tokenizer
        n = rng.randint(3, self.N)

        # Build random permutation (directed cycle cover)
        nodes = list(range(n))
        rng.shuffle(nodes)
        perm = {}  # perm[x] = y means x -> y
        for i in range(n):
            perm[nodes[i]] = nodes[(i + 1) % n]

        # Encode edges in random order
        edges = list(perm.items())
        rng.shuffle(edges)

        tokens = [tok.BOS]
        for x, y in edges:
            tokens += tok.encode_node(x)
            tokens += tok.encode_node(y)

        # Generate queries
        for _ in range(self.n_queries):
            k = rng.randint(1, self.K)
            q = rng.choice(nodes)

            # Compute k-th successor
            cur = q
            for _ in range(k):
                cur = perm[cur]

            tokens.append(tok.query_token(k))
            tokens += tok.encode_node(q)
            tokens += tok.encode_node(cur)

        tokens.append(tok.EOS)
        return tokens

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        seed = self.seed + (worker_info.id if worker_info else 0)
        rng = random.Random(seed)

        buffer = []
        while True:
            inst = self._make_instance(rng)
            buffer.extend(inst)
            while len(buffer) >= self.context_len:
                chunk = buffer[:self.context_len]
                buffer = buffer[self.context_len:]
                yield torch.tensor(chunk, dtype=torch.long)


def build_depo_dataset(variant="depo1", N=225, K=8, context_len=2048, seed=42):
    return DepoDataset(N, K, variant, context_len, seed=seed)
