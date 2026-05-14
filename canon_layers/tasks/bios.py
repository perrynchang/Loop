"""
Task BioS: Knowledge Storage and Extraction
Synthetic biography dataset based on "Physics of Language Models: Part 3.1"
(Zhu & Li, 2023).

Each individual has 6 attributes: birth_month, birth_city, university, major,
company, and company_city (derived from company via a fixed mapping).

The core finding: models trained on fixed-order biographies memorize but cannot
extract knowledge via QA (≈0% OOD accuracy); shuffling the 6 attribute sentences
(permute augmentation) enables knowledge extraction (>70% accuracy).

Sequence format (mixed pretraining):
  [BOS] [PERSON_i]
  ([PERSON_i]) [BIO_MARK] [value]   ×6 blocks, optionally shuffled + repeated
  [QUERY] [Q_MARK] [PERSON_i] [answer]   ×n_queries
  [EOS]

Augmentations via `augment` string (combine with '+' or ','):
  permute    shuffle the 6 bio attribute blocks each instance
  fullname   prepend person token before each attribute block
  multi<M>   repeat bio M times with independent shuffles per instance (e.g. multi2)
"""
import re
import random
import torch
from torch.utils.data import IterableDataset


ATTR_NAMES = ['month', 'city', 'univ', 'major', 'company', 'ccity']


class BioSTokenizer:
    """
    Integer vocabulary for BioS. All values are non-negative token IDs.

    Layout:
      0          BOS
      1          EOS
      2          QUERY
      3..8       bio attribute markers  (one per attribute in ATTR_NAMES order)
      9..14      query attribute markers
      15..14+N   person tokens
      15+N..     attribute value tokens (month, city, univ, major, company, ccity)
    """

    def __init__(self, N=1000, n_cities=200, n_univs=300, n_majors=100, n_companies=263):
        self.N = N
        self.attr_sizes = dict(
            month=12, city=n_cities, univ=n_univs,
            major=n_majors, company=n_companies,
            ccity=n_companies,  # 1:1 mapping company → company_city
        )

        self.BOS = 0
        self.EOS = 1
        self.QUERY = 2

        # Bio and query attribute markers
        self.bio_marks   = {name: 3 + i                    for i, name in enumerate(ATTR_NAMES)}
        self.query_marks = {name: 3 + len(ATTR_NAMES) + i  for i, name in enumerate(ATTR_NAMES)}

        n_special = 3 + 2 * len(ATTR_NAMES)  # = 15

        self.person_offset = n_special
        offset = self.person_offset + N

        self.attr_offsets = {}
        for name in ATTR_NAMES:
            self.attr_offsets[name] = offset
            offset += self.attr_sizes[name]

        self.total_vocab = offset

    def person_tok(self, pid):
        return self.person_offset + pid

    def attr_tok(self, name, value):
        return self.attr_offsets[name] + value

    def bio_mark(self, name):
        return self.bio_marks[name]

    def query_mark(self, name):
        return self.query_marks[name]


class BioSDataset(IterableDataset):
    """
    Mixed-training BioS dataset: chunks contain packed biography + QA sequences.

    Parameters
    ----------
    N           : number of individuals (default 1000; paper uses 100 000)
    augment     : augmentation string, e.g. "permute+fullname+multi2"
    n_queries   : QA pairs appended per bio instance
    context_len : tokens per yielded training chunk
    n_cities / n_univs / n_majors / n_companies : attribute vocabulary sizes
    seed        : RNG seed (instance generation is deterministic per-person)
    """

    def __init__(
        self,
        N=1000,
        augment="",
        n_queries=4,
        context_len=512,
        n_cities=200,
        n_univs=300,
        n_majors=100,
        n_companies=263,
        seed=42,
    ):
        super().__init__()
        self.tok = BioSTokenizer(N, n_cities, n_univs, n_majors, n_companies)
        self.N = N
        self.n_queries = n_queries
        self.context_len = context_len
        self.seed = seed

        # Parse augment string
        parts = {p.strip().lower() for p in re.split(r'[,+]', augment) if p.strip()}
        self.do_permute  = 'permute'   in parts
        self.do_fullname = 'fullname'  in parts
        self.multi_M = 1
        for part in parts:
            m = re.match(r'multi(\d+)$', part)
            if m:
                self.multi_M = int(m.group(1))

        # Pre-generate all N individuals' attributes with a fixed seed
        rng = random.Random(seed)
        self.persons = []
        for _ in range(N):
            attrs = {}
            for name in ATTR_NAMES:
                if name == 'ccity':
                    # company_city is deterministically derived from company
                    attrs['ccity'] = attrs['company'] % self.tok.attr_sizes['ccity']
                else:
                    attrs[name] = rng.randrange(self.tok.attr_sizes[name])
            self.persons.append(attrs)

    def _make_instance(self, pid, rng):
        tok   = self.tok
        p_tok = tok.person_tok(pid)
        attrs = self.persons[pid]

        tokens = [tok.BOS, p_tok]

        # Biography: multi_M passes, each possibly with a different shuffle
        for _ in range(self.multi_M):
            blocks = [(tok.bio_mark(name), tok.attr_tok(name, attrs[name]))
                      for name in ATTR_NAMES]
            if self.do_permute:
                rng.shuffle(blocks)
            for mark, value in blocks:
                if self.do_fullname:
                    tokens.append(p_tok)
                tokens.append(mark)
                tokens.append(value)

        # QA queries
        for name in rng.choices(ATTR_NAMES, k=self.n_queries):
            tokens += [tok.QUERY, tok.query_mark(name), p_tok,
                       tok.attr_tok(name, attrs[name])]

        tokens.append(tok.EOS)
        return tokens

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        seed = self.seed + (worker_info.id if worker_info else 0)
        rng  = random.Random(seed)

        buffer = []
        while True:
            pid = rng.randrange(self.N)
            buffer.extend(self._make_instance(pid, rng))
            while len(buffer) >= self.context_len:
                yield torch.tensor(buffer[:self.context_len], dtype=torch.long)
                buffer = buffer[self.context_len:]


def build_bios_dataset(N=1000, augment="", n_queries=4, context_len=512, seed=42, **kwargs):
    return BioSDataset(N=N, augment=augment, n_queries=n_queries,
                       context_len=context_len, seed=seed, **kwargs)


# ---------------------------------------------------------------------------
# Part 3.2 – Knowledge Manipulation
# ---------------------------------------------------------------------------

# Query type groups for Part 3.2
CLASSIFY_TYPES = ['month_even', 'month_mod6', 'major_lucky', 'major_lucky_mod5']
COMPARE_TYPES  = ['month_rank', 'month_diff', 'major_lucky_rank', 'major_lucky_diff']
INVERSE_TYPES  = ['inv_month', 'inv_city', 'inv_univ']
ALL_MANIP_TYPES = CLASSIFY_TYPES + COMPARE_TYPES + INVERSE_TYPES


def _parse_augment(augment):
    """Return (do_permute, do_fullname, multi_M, reverse_pos) from augment string."""
    parts = {p.strip().lower() for p in re.split(r'[,+]', augment) if p.strip()}
    do_permute  = 'permute'  in parts
    do_fullname = 'fullname' in parts
    multi_M = 1
    reverse_pos = 0
    for p in parts:
        m = re.match(r'multi(\d+)$', p)
        if m:
            multi_M = int(m.group(1))
        m = re.match(r'reverse(\d+)$', p)
        if m:
            reverse_pos = int(m.group(1))
    return do_permute, do_fullname, multi_M, reverse_pos


class BioS32Tokenizer(BioSTokenizer):
    """
    Extended vocabulary for Part 3.2 manipulation tasks.

    Appended after the BioSTokenizer layout:
      YES / NO / COT_SEP
      Classification query markers  (4)
      Comparison query markers       (4)
      Inverse-search query markers   (3)
      mod6 answer tokens             (6)
      luckiness answer tokens        (n_majors)
      mod5 answer tokens             (5)
      month-diff tokens              (23, for differences -11..11)
      major-diff tokens              (2*n_majors-1, for differences -(n-1)..(n-1))
    """

    def __init__(self, N=1000, n_cities=200, n_univs=300, n_majors=100, n_companies=263):
        super().__init__(N, n_cities, n_univs, n_majors, n_companies)
        b = self.total_vocab

        # Special answer / CoT tokens
        self.YES     = b; b += 1
        self.NO      = b; b += 1
        self.COT_SEP = b; b += 1

        # Classification query markers
        self.Q_MONTH_EVEN       = b; b += 1
        self.Q_MONTH_MOD6       = b; b += 1
        self.Q_MAJOR_LUCKY      = b; b += 1
        self.Q_MAJOR_LUCKY_MOD5 = b; b += 1

        # Comparison query markers
        self.Q_MONTH_RANK       = b; b += 1
        self.Q_MONTH_DIFF       = b; b += 1
        self.Q_MAJOR_LUCKY_RANK = b; b += 1
        self.Q_MAJOR_LUCKY_DIFF = b; b += 1

        # Inverse-search query markers
        self.Q_INV_MONTH = b; b += 1
        self.Q_INV_CITY  = b; b += 1
        self.Q_INV_UNIV  = b; b += 1

        # Answer value token ranges
        self.mod6_offset       = b; b += 6               # month % 6 → 0..5
        self.lucky_offset      = b; b += n_majors         # luckiness → 0..n_majors-1
        self.mod5_offset       = b; b += 5               # luckiness % 5 → 0..4
        self.month_diff_offset = b; b += 23              # month_A - month_B → -11..11
        self.major_diff_offset = b; b += 2 * n_majors - 1  # lucky_A - lucky_B

        self.total_vocab = b

        # Fixed luckiness mapping: major_id → score in 0..n_majors-1
        _rng = random.Random(12345)
        self.major_luckiness = list(range(n_majors))
        _rng.shuffle(self.major_luckiness)

    # -- answer token helpers ------------------------------------------------

    def lucky_tok(self, major_id):
        return self.lucky_offset + self.major_luckiness[major_id]

    def mod6_tok(self, month):
        return self.mod6_offset + (month % 6)

    def mod5_tok(self, major_id):
        return self.mod5_offset + (self.major_luckiness[major_id] % 5)

    def month_diff_tok(self, diff):         # diff in [-11, 11]
        return self.month_diff_offset + diff + 11

    def major_diff_tok(self, diff):         # diff in [-(n-1), n-1]
        return self.major_diff_offset + diff + (self.attr_sizes['major'] - 1)


class BioS32Dataset(IterableDataset):
    """
    Part 3.2 dataset: biography sequences with classification, comparison, and
    inverse-search queries, optionally with Chain-of-Thought (CoT) hints.

    Parameters (beyond BioSDataset)
    --------------------------------
    query_types : 'all' | 'classify' | 'compare' | 'inverse' | 'extract'
                  or an explicit list of type strings.
    cot_prob    : probability of including CoT hint tokens for manipulation queries.
    augment     : same as BioSDataset, plus 'reverse<N>' to move person name after
                  the N-th attribute block (enables inverse search; reverse6 = name last).
    """

    def __init__(
        self,
        N=1000,
        augment="permute",
        n_queries=4,
        context_len=512,
        n_cities=200,
        n_univs=300,
        n_majors=100,
        n_companies=263,
        seed=42,
        query_types='all',
        cot_prob=0.5,
    ):
        super().__init__()
        self.tok = BioS32Tokenizer(N, n_cities, n_univs, n_majors, n_companies)
        self.N = N
        self.n_queries = n_queries
        self.context_len = context_len
        self.seed = seed
        self.cot_prob = cot_prob

        self.do_permute, self.do_fullname, self.multi_M, self.reverse_pos = \
            _parse_augment(augment)

        # Resolve active query type list
        _type_map = {
            'classify': CLASSIFY_TYPES,
            'compare':  COMPARE_TYPES,
            'inverse':  INVERSE_TYPES,
        }
        if query_types == 'all':
            self.active_queries = ['extract'] + ALL_MANIP_TYPES
        elif query_types == 'extract':
            self.active_queries = ['extract']
        elif query_types in _type_map:
            self.active_queries = ['extract'] + _type_map[query_types]
        elif isinstance(query_types, list):
            self.active_queries = query_types
        else:
            self.active_queries = [query_types]

        # Pre-generate all N individuals (same convention as BioSDataset)
        rng = random.Random(seed)
        self.persons = []
        for _ in range(N):
            attrs = {}
            for name in ATTR_NAMES:
                if name == 'ccity':
                    attrs['ccity'] = attrs['company'] % self.tok.attr_sizes['ccity']
                else:
                    attrs[name] = rng.randrange(self.tok.attr_sizes[name])
            self.persons.append(attrs)

    # -- bio generation ------------------------------------------------------

    def _bio_tokens(self, pid, rng):
        """Build the bio token list for one person (multi_M passes)."""
        tok   = self.tok
        p_tok = tok.person_tok(pid)
        attrs = self.persons[pid]
        result = []

        for _ in range(self.multi_M):
            blocks = [(tok.bio_mark(name), tok.attr_tok(name, attrs[name]))
                      for name in ATTR_NAMES]
            if self.do_permute:
                rng.shuffle(blocks)

            entry = [tok.BOS]
            if self.reverse_pos == 0:
                entry.append(p_tok)

            for i, (mark, value) in enumerate(blocks):
                if self.do_fullname and self.reverse_pos == 0:
                    entry.append(p_tok)
                entry.append(mark)
                entry.append(value)
                # Insert name after the i+1-th block for reverse augmentations
                if self.reverse_pos > 0 and i + 1 == self.reverse_pos:
                    entry.append(p_tok)

            # reverse_pos ≥ len(blocks) → name at end
            if self.reverse_pos >= len(blocks):
                entry.append(p_tok)

            result.extend(entry)
        return result

    # -- query generation ----------------------------------------------------

    def _classify_tokens(self, pid, q_type, rng):
        tok   = self.tok
        p_tok = tok.person_tok(pid)
        attrs = self.persons[pid]
        do_cot = rng.random() < self.cot_prob

        Q_MARK = {
            'month_even':       tok.Q_MONTH_EVEN,
            'month_mod6':       tok.Q_MONTH_MOD6,
            'major_lucky':      tok.Q_MAJOR_LUCKY,
            'major_lucky_mod5': tok.Q_MAJOR_LUCKY_MOD5,
        }
        tokens = [tok.QUERY, Q_MARK[q_type], p_tok]

        if q_type == 'month_even':
            if do_cot:
                tokens += [tok.attr_tok('month', attrs['month']), tok.COT_SEP]
            tokens.append(tok.YES if attrs['month'] % 2 == 0 else tok.NO)
        elif q_type == 'month_mod6':
            if do_cot:
                tokens += [tok.attr_tok('month', attrs['month']), tok.COT_SEP]
            tokens.append(tok.mod6_tok(attrs['month']))
        elif q_type == 'major_lucky':
            if do_cot:
                tokens += [tok.attr_tok('major', attrs['major']), tok.COT_SEP]
            tokens.append(tok.lucky_tok(attrs['major']))
        elif q_type == 'major_lucky_mod5':
            if do_cot:
                tokens += [tok.attr_tok('major', attrs['major']), tok.COT_SEP]
            tokens.append(tok.mod5_tok(attrs['major']))
        return tokens

    def _compare_tokens(self, pid_a, pid_b, q_type, rng):
        tok    = self.tok
        pa     = tok.person_tok(pid_a)
        pb     = tok.person_tok(pid_b)
        attrs_a = self.persons[pid_a]
        attrs_b = self.persons[pid_b]
        do_cot  = rng.random() < self.cot_prob

        Q_MARK = {
            'month_rank':       tok.Q_MONTH_RANK,
            'month_diff':       tok.Q_MONTH_DIFF,
            'major_lucky_rank': tok.Q_MAJOR_LUCKY_RANK,
            'major_lucky_diff': tok.Q_MAJOR_LUCKY_DIFF,
        }
        tokens = [tok.QUERY, Q_MARK[q_type], pa, pb]

        if q_type == 'month_rank':
            if do_cot:
                tokens += [tok.attr_tok('month', attrs_a['month']),
                           tok.attr_tok('month', attrs_b['month']), tok.COT_SEP]
            tokens.append(tok.YES if attrs_a['month'] > attrs_b['month'] else tok.NO)
        elif q_type == 'month_diff':
            if do_cot:
                tokens += [tok.attr_tok('month', attrs_a['month']),
                           tok.attr_tok('month', attrs_b['month']), tok.COT_SEP]
            tokens.append(tok.month_diff_tok(attrs_a['month'] - attrs_b['month']))
        elif q_type == 'major_lucky_rank':
            if do_cot:
                tokens += [tok.lucky_tok(attrs_a['major']),
                           tok.lucky_tok(attrs_b['major']), tok.COT_SEP]
            la = tok.major_luckiness[attrs_a['major']]
            lb = tok.major_luckiness[attrs_b['major']]
            tokens.append(tok.YES if la > lb else tok.NO)
        elif q_type == 'major_lucky_diff':
            if do_cot:
                tokens += [tok.lucky_tok(attrs_a['major']),
                           tok.lucky_tok(attrs_b['major']), tok.COT_SEP]
            diff = tok.major_luckiness[attrs_a['major']] - tok.major_luckiness[attrs_b['major']]
            tokens.append(tok.major_diff_tok(diff))
        return tokens

    def _inverse_tokens(self, pid, q_type):
        tok       = self.tok
        attr_name = q_type[4:]          # strip 'inv_'
        attr_val  = self.persons[pid][attr_name]
        Q_MARK = {
            'inv_month': tok.Q_INV_MONTH,
            'inv_city':  tok.Q_INV_CITY,
            'inv_univ':  tok.Q_INV_UNIV,
        }
        return [tok.QUERY, Q_MARK[q_type],
                tok.attr_tok(attr_name, attr_val),
                tok.person_tok(pid)]

    # -- instance assembly ---------------------------------------------------

    def _make_instance(self, pid, rng):
        tok   = self.tok
        p_tok = tok.person_tok(pid)

        tokens = self._bio_tokens(pid, rng)

        # Pick person B for comparison queries; include their bio in context
        pid_b = rng.randrange(self.N)
        while pid_b == pid:
            pid_b = rng.randrange(self.N)
        if any(q in self.active_queries for q in COMPARE_TYPES):
            tokens += self._bio_tokens(pid_b, rng)

        for _ in range(self.n_queries):
            q_type = rng.choice(self.active_queries)
            if q_type == 'extract':
                attr = rng.choice(ATTR_NAMES)
                tokens += [tok.QUERY, tok.query_mark(attr), p_tok,
                           tok.attr_tok(attr, self.persons[pid][attr])]
            elif q_type in CLASSIFY_TYPES:
                tokens += self._classify_tokens(pid, q_type, rng)
            elif q_type in COMPARE_TYPES:
                tokens += self._compare_tokens(pid, pid_b, q_type, rng)
            elif q_type in INVERSE_TYPES:
                tokens += self._inverse_tokens(pid, q_type)

        tokens.append(tok.EOS)
        return tokens

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        seed = self.seed + (worker_info.id if worker_info else 0)
        rng  = random.Random(seed)

        buffer = []
        while True:
            pid = rng.randrange(self.N)
            buffer.extend(self._make_instance(pid, rng))
            while len(buffer) >= self.context_len:
                yield torch.tensor(buffer[:self.context_len], dtype=torch.long)
                buffer = buffer[self.context_len:]


def build_bios32_dataset(N=1000, augment="permute", n_queries=4, context_len=512,
                          seed=42, query_types='all', cot_prob=0.5, **kwargs):
    return BioS32Dataset(N=N, augment=augment, n_queries=n_queries,
                         context_len=context_len, seed=seed,
                         query_types=query_types, cot_prob=cot_prob, **kwargs)
