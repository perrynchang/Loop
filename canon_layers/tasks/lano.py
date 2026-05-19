"""
Task Lano: Hierarchical Language Structure
Context-free grammar (CFG) datasets: cfg3f, cfg3j, cfg3k.
Models must learn to generate valid CFG sentences without explicit rules.

Grammar (cfg3f) starting from NT 22:
  22 -> 20 21 | 20 19 21 | 21 19 19 | 20 20

NT symbols: 7..22  (integers)
T symbols: 1, 2, 3
"""
import random
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import IterableDataset


# CFG definitions for each dataset variant
# Each rule: NT -> list of (NT or T)
# T symbols: 1, 2, 3
# NT symbols are integers >= 7

CFG3F_RULES: Dict[int, List[List[int]]] = {
    22: [[20, 21], [20, 19, 21], [21, 19, 19], [20, 20]],
    21: [[19, 19], [19, 20], [19]],
    20: [[19], [19, 19]],
    19: [[1], [2], [3]],
}

# cfg3j extends cfg3f by one level (intermediate depth)
CFG3J_RULES: Dict[int, List[List[int]]] = {
    25: [[22, 22], [22, 23, 22], [23, 22, 22]],
    23: [[22], [21]],
    22: [[20, 21], [20, 19, 21], [21, 19, 19], [20, 20]],
    21: [[19, 19], [19, 20], [19]],
    20: [[19], [19, 19]],
    19: [[1], [2], [3]],
}

# cfg3k extends cfg3f by one level (deeper recursion, longer sequences)
CFG3K_RULES: Dict[int, List[List[int]]] = {
    25: [[22, 22], [22, 23, 22], [23, 22, 22], [22, 22, 23]],
    24: [[22, 21], [21, 22]],
    23: [[22], [21], [20]],
    22: [[20, 21], [20, 19, 21], [21, 19, 19], [20, 20]],
    21: [[19, 19], [19, 20], [19]],
    20: [[19], [19, 19]],
    19: [[1], [2], [3]],
}

TERMINAL_SYMBOLS = {1, 2, 3}
CFG_ROOTS = {"cfg3f": 22, "cfg3j": 25, "cfg3k": 25}
CFG_RULES = {"cfg3f": CFG3F_RULES, "cfg3j": CFG3J_RULES, "cfg3k": CFG3K_RULES}


def generate_sentence(rules, root, rng, max_len=2000):
    """Generate a random sentence from the CFG by expanding the root symbol."""
    stack = [root]
    tokens = []
    steps = 0
    while stack and steps < max_len:
        sym = stack.pop(0)
        if sym in TERMINAL_SYMBOLS:
            tokens.append(sym)
        elif sym in rules:
            expansion = rng.choice(rules[sym])
            stack = expansion + stack
        steps += 1
    if stack:
        return None  # failed / too long
    return tokens


def is_valid_cfg(tokens, rules, root):
    """
    Validate a token sequence against a CFG.
    Returns True if the sequence can be derived from root.
    Uses full backtracking to handle ambiguous grammars correctly.
    """
    def parse(sym, pos):
        """Return the set of all possible end positions after parsing sym from pos."""
        if sym in TERMINAL_SYMBOLS:
            if pos < len(tokens) and tokens[pos] == sym:
                return {pos + 1}
            return set()
        if sym not in rules:
            return set()
        ends = set()
        for expansion in rules[sym]:
            positions = {pos}
            for s in expansion:
                next_positions = set()
                for p in positions:
                    next_positions |= parse(s, p)
                positions = next_positions
                if not positions:
                    break
            ends |= positions
        return ends

    return len(tokens) in parse(root, 0)


class LanoTokenizer:
    def __init__(self):
        self.BOS = 0
        # Tokens 1-25 map directly to CFG symbols (1,2,3 terminals + NT symbols)
        self.SYMBOL_OFFSET = 0  # symbol s has token s (1..25)
        self.total_vocab = 26  # 0=BOS, 1-25=CFG symbols (3 terminals + NTs)


class LanoDataset(IterableDataset):
    """
    Generates Lano task sequences: concatenated CFG sentences separated by BOS tokens.
    Training: predict next token given context.
    The model learns the CFG structure implicitly.
    """

    def __init__(self, variant="cfg3f", context_len=512, seed=42):
        super().__init__()
        self.variant = variant
        self.rules = CFG_RULES[variant]
        self.root = CFG_ROOTS[variant]
        self.context_len = context_len
        self.seed = seed
        self.tokenizer = LanoTokenizer()

    def _make_sentence(self, rng):
        sentence = None
        while sentence is None:
            sentence = generate_sentence(self.rules, self.root, rng)
        return sentence

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        seed = self.seed + (worker_info.id if worker_info else 0)
        rng = random.Random(seed)

        buffer = [self.tokenizer.BOS]
        while True:
            sentence = self._make_sentence(rng)
            buffer.extend(sentence)
            buffer.append(self.tokenizer.BOS)  # separator

            while len(buffer) >= self.context_len:
                chunk = buffer[:self.context_len]
                buffer = buffer[self.context_len:]
                yield torch.tensor(chunk, dtype=torch.long)


def evaluate_lano(model, tokenizer, rules, root, n_samples=200, max_gen_len=500, device='cpu'):
    """
    Evaluate model by generating sentences from BOS and checking CFG validity.
    Uses temperature 1 sampling.
    """
    correct = 0
    model.eval()
    with torch.no_grad():
        for _ in range(n_samples):
            # Start from BOS
            x = torch.tensor([[tokenizer.BOS]], dtype=torch.long, device=device)
            generated = []
            for _ in range(max_gen_len):
                logits = model(x)
                probs = torch.softmax(logits[0, -1], dim=-1)
                next_tok = torch.multinomial(probs, 1).item()
                if next_tok == tokenizer.BOS:
                    break
                generated.append(next_tok)
                x = torch.cat([x, torch.tensor([[next_tok]], device=device)], dim=1)

            if generated and is_valid_cfg(generated, rules, root):
                correct += 1

    return correct / n_samples


def build_lano_dataset(variant="cfg3f", seed=42):
    context_len = 512 if variant == "cfg3f" else 1536
    return LanoDataset(variant, context_len=context_len, seed=seed)
