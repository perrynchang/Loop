"""
Task Depo: Mental Reasoning Depth
k-hop traversal over directed permutations.
Format: <bos> x1 y1 x2 y2 ... xn yn <query_k1> q1 <ans> a1 <query_k2> q2 <ans> a2 ...
"""
import math
import random
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset


class DepoTokenizer:
    """Simple tokenizer for Depo task."""

    def __init__(self, variant="depo1"):
        self.variant = variant
        if variant == "depo1":
            self.vocab_size = 50
            self.node_min_len = 1
            self.node_max_len = 2
        else:  # depo2 — word-boundary encoding doubles effective node vocab
            self.vocab_size = 4
            self.node_min_len = 5
            self.node_max_len = 7

        # Special tokens: <bos>=0, <ans>=1, then node vocab, then <query_k> tokens
        self.BOS = 0
        self.ANS = 1
        self.NODE_VOCAB_OFFSET = 2
        # Both depo1 and depo2 use word-boundary encoding: 2*vocab_size node token IDs
        self.K_MAX = 16
        self.QUERY_OFFSET = self.NODE_VOCAB_OFFSET + self.vocab_size * 2
        self.total_vocab = self.QUERY_OFFSET + self.K_MAX + 1

    def encode_node(self, node_id):
        """Encode a node as a sequence of tokens.

        Both length and content are seeded from node_id so every call with the
        same node_id returns the identical token sequence within one instance.
        Word-boundary encoding: inner tokens from [0, V-1], final token from [V, 2V-1].
        """
        rng = random.Random(node_id)
        length = rng.randint(self.node_min_len, self.node_max_len)
        if length > 1:
            toks = [self.NODE_VOCAB_OFFSET + rng.randint(0, self.vocab_size - 1)
                    for _ in range(length - 1)]
            toks.append(self.NODE_VOCAB_OFFSET + self.vocab_size +
                        rng.randint(0, self.vocab_size - 1))
            return toks
        # length == 1: single token acts as its own boundary token
        return [self.NODE_VOCAB_OFFSET + self.vocab_size + rng.randint(0, self.vocab_size - 1)]

    def query_token(self, k):
        return self.QUERY_OFFSET + k


class DepoDataset(IterableDataset):
    """
    Generates Depo task instances on-the-fly.

    Each instance:
      <bos> x1_toks y1_toks ... xn_toks yn_toks
      <query_k1> q1_toks <ans> a1_toks ... (t = min(10, n) queries)

    Instances are concatenated and left-aligned into context_len windows.
    Yields (tokens, answer_mask) where answer_mask=1 for <ans> and answer tokens.
    """

    def __init__(self, N, K, variant="depo1", context_len=2048, seed=42, rank=0):
        super().__init__()
        self.N = N
        self.K = K
        self.tokenizer = DepoTokenizer(variant)
        self.context_len = context_len
        self.seed = seed + rank * 10000

        # Precompute CDF for curriculum sampling n ∝ 1/(n + √N), matching author's distribution
        ns = list(range(3, N + 1))
        w = [1.0 / (n + math.sqrt(N) + 1e-12) for n in ns]
        total = sum(w)
        self._sample_ns = ns
        self._sample_cdf = []
        s = 0.0
        for wi in w:
            s += wi / total
            self._sample_cdf.append(s)

    def _sample_n(self, rng):
        """Sample n ∝ 1/(n+√N) via binary search on precomputed CDF."""
        r = rng.random()
        lo, hi = 0, len(self._sample_cdf) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._sample_cdf[mid] < r:
                lo = mid + 1
            else:
                hi = mid
        return self._sample_ns[lo]

    def _make_instance(self, rng):
        """Return (tokens, answer_mask) for one problem instance."""
        tok = self.tokenizer
        n = self._sample_n(rng)

        # Build random permutation (single directed cycle)
        nodes = list(range(n))
        rng.shuffle(nodes)
        perm = {nodes[i]: nodes[(i + 1) % n] for i in range(n)}

        # Encode edges in random order
        edges = list(perm.items())
        rng.shuffle(edges)

        tokens = [tok.BOS]
        mask = [0]
        for x, y in edges:
            xt, yt = tok.encode_node(x), tok.encode_node(y)
            tokens += xt + yt
            mask += [0] * (len(xt) + len(yt))

        # t = min(10, n) queries per instance
        for _ in range(min(10, n)):
            k = rng.randint(1, self.K)
            q = rng.choice(nodes)
            cur = q
            for _ in range(k):
                cur = perm[cur]

            q_toks = tok.encode_node(q)
            a_toks = tok.encode_node(cur)
            tokens += [tok.query_token(k)] + q_toks + [tok.ANS] + a_toks
            mask += [0] * (1 + len(q_toks))   # query token + query node: not in loss
            mask += [1] + [1] * len(a_toks)    # <ans> + answer tokens: in loss

        return tokens, mask

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        seed = self.seed + (worker_info.id if worker_info else 0)
        rng = random.Random(seed)
        tok = self.tokenizer

        token_buf, mask_buf = [], []
        while True:
            toks, mask = self._make_instance(rng)
            token_buf.extend(toks)
            mask_buf.extend(mask)

            while len(token_buf) >= self.context_len:
                chunk_toks = token_buf[:self.context_len]
                chunk_mask = mask_buf[:self.context_len]
                rest_toks = token_buf[self.context_len:]
                rest_mask = mask_buf[self.context_len:]

                # Left-align next chunk: discard partial instance tail up to next BOS
                next_bos = next((i for i, t in enumerate(rest_toks) if t == tok.BOS), None)
                if next_bos is None:
                    token_buf, mask_buf = [], []
                else:
                    token_buf = rest_toks[next_bos:]
                    mask_buf = rest_mask[next_bos:]

                yield (
                    torch.tensor(chunk_toks, dtype=torch.long),
                    torch.tensor(chunk_mask, dtype=torch.bool),
                )


def build_depo_dataset(variant="depo1", N=225, K=8, context_len=2048, seed=42, rank=0):
    return DepoDataset(N, K, variant, context_len, seed=seed, rank=rank)
