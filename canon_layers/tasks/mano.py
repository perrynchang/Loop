"""
Task Mano: Knowledge Manipulation
Multi-step modular arithmetic (mod 23) in prefix notation.
Models must learn 23x23 operation tables and compose them hierarchically.
Format: <bos> <len_L> op1 op2 a b - c d <ans> result <eos>
"""
import random
import torch
from torch.utils.data import IterableDataset

MOD = 23
OPS = ['+', '-', '*']


def eval_expr(tree):
    """Evaluate a prefix expression tree. tree is (op, left, right) or int."""
    if isinstance(tree, int):
        return tree
    op, left, right = tree
    l = eval_expr(left) % MOD
    r = eval_expr(right) % MOD
    if op == '+':
        return (l + r) % MOD
    elif op == '-':
        return (l - r) % MOD
    else:
        return (l * r) % MOD


def build_expr(length, rng):
    """Build a random prefix expression tree with `length` operations."""
    if length == 0:
        return rng.randint(0, MOD - 1)
    op = rng.choice(OPS)
    # Split remaining length-1 ops between left and right subtrees
    left_len = rng.randint(0, length - 1)
    right_len = length - 1 - left_len
    return (op, build_expr(left_len, rng), build_expr(right_len, rng))


def serialize_expr(tree):
    """Convert expression tree to list of string tokens."""
    if isinstance(tree, int):
        return [str(tree)]
    op, left, right = tree
    return [op] + serialize_expr(left) + serialize_expr(right)


class ManoTokenizer:
    def __init__(self):
        # Tokens: BOS, EOS, ANS, len_1..len_L_max, ops +,-,*, values 0..22
        self.BOS = 0
        self.EOS = 1
        self.ANS = 2
        self.L_MAX = 16
        self.LEN_OFFSET = 3          # len_1 = 3, len_2 = 4, ...
        self.OP_OFFSET = 3 + self.L_MAX  # +, -, *
        self.VAL_OFFSET = self.OP_OFFSET + 3  # 0..22
        self.total_vocab = self.VAL_OFFSET + MOD

        self.op_to_id = {'+': 0, '-': 1, '*': 2}

    def len_token(self, l):
        return self.LEN_OFFSET + l - 1

    def op_token(self, op):
        return self.OP_OFFSET + self.op_to_id[op]

    def val_token(self, v):
        return self.VAL_OFFSET + v

    def encode_expr(self, expr_tokens):
        result = []
        for t in expr_tokens:
            if t in ('+', '-', '*'):
                result.append(self.op_token(t))
            else:
                result.append(self.val_token(int(t)))
        return result


class ManoDataset(IterableDataset):
    """
    Generates Mano task instances.
    Each: <bos> <len_l> [prefix_expr_tokens] <ans> [result_token] <eos>
    Length l is sampled uniformly from [1, L].
    """

    def __init__(self, L, context_len=1024, seed=42):
        super().__init__()
        self.L = L
        self.tokenizer = ManoTokenizer()
        self.context_len = context_len
        self.seed = seed

    def _make_instance(self, rng):
        tok = self.tokenizer
        l = rng.randint(1, self.L)
        tree = build_expr(l, rng)
        result = eval_expr(tree) % MOD
        expr_tokens = serialize_expr(tree)

        tokens = [tok.BOS, tok.len_token(l)]
        tokens += tok.encode_expr(expr_tokens)
        tokens.append(tok.ANS)
        tokens.append(tok.val_token(result))
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


def evaluate_mano(model, tokenizer, L, n_samples=500, device='cpu'):
    """Evaluate model accuracy on Mano task at max difficulty l=L."""
    rng = random.Random(12345)
    correct = 0
    total = 0
    tok = tokenizer

    model.eval()
    with torch.no_grad():
        for _ in range(n_samples):
            l = L
            tree = build_expr(l, rng)
            expected = eval_expr(tree) % MOD
            expr_tokens = serialize_expr(tree)

            prefix = [tok.BOS, tok.len_token(l)] + tok.encode_expr(expr_tokens) + [tok.ANS]
            x = torch.tensor([prefix], dtype=torch.long, device=device)
            logits = model(x)
            pred = logits[0, -1].argmax().item()
            if pred == tok.val_token(expected):
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0


def build_mano_dataset(L=10, context_len=1024, seed=42):
    return ManoDataset(L, context_len, seed=seed)
