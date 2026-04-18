"""
Task Capo: Knowledge Capacity
Synthetic biographies (bioS dataset) for measuring bits-per-parameter knowledge storage.
Each biography has 6 attributes; models are trained for 100 exposures each.
"""
import random
from typing import List
import torch
from torch.utils.data import Dataset

# Attribute pools (simplified; paper uses larger pools)
FIRST_NAMES = [
    "Anya", "Blake", "Casey", "Dana", "Ellis", "Faye", "Grey", "Harper",
    "Iris", "Jordan", "Kim", "Lee", "Morgan", "Noah", "Owen", "Parker",
    "Quinn", "River", "Sage", "Taylor", "Uma", "Vale", "Wren", "Xen", "Yael",
]
MIDDLE_NAMES = ["Briar", "Cedar", "Dawn", "Echo", "Frost", "Glen", "Haze", "Ivy"]
LAST_NAMES = [
    "Forger", "Archer", "Blake", "Chase", "Drake", "Ellis", "Fox", "Grant",
    "Hayes", "Innis", "James", "Knox", "Lane", "Marsh", "Nash", "Oaks",
]

BIRTH_MONTHS = ["January", "February", "March", "April", "May", "June",
                 "July", "August", "September", "October", "November", "December"]
BIRTH_DAYS = list(range(1, 29))
BIRTH_YEARS = list(range(1950, 2005))
CITIES = [
    "New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX",
    "Princeton, NJ", "Austin, TX", "Seattle, WA", "Boston, MA",
    "Denver, CO", "Atlanta, GA", "Miami, FL", "Phoenix, AZ",
]
UNIVERSITIES = [
    "Massachusetts Institute of Technology", "Stanford University",
    "Harvard University", "Yale University", "Princeton University",
    "Columbia University", "University of Chicago", "Duke University",
    "Cornell University", "Johns Hopkins University",
]
MAJORS = [
    "Computer Science", "Mathematics", "Physics", "Chemistry", "Biology",
    "Economics", "Communications", "English Literature", "Philosophy",
    "Engineering", "Psychology", "Political Science",
]
EMPLOYERS = [
    "Meta Platforms", "Google", "Apple", "Microsoft", "Amazon",
    "Netflix", "Twitter", "Uber", "Airbnb", "Spotify",
    "OpenAI", "DeepMind", "Anthropic", "Nvidia", "Intel",
]
EMPLOYER_CITIES = {
    "Meta Platforms": "Menlo Park, CA",
    "Google": "Mountain View, CA",
    "Apple": "Cupertino, CA",
    "Microsoft": "Redmond, WA",
    "Amazon": "Seattle, WA",
    "Netflix": "Los Gatos, CA",
    "Twitter": "San Francisco, CA",
    "Uber": "San Francisco, CA",
    "Airbnb": "San Francisco, CA",
    "Spotify": "New York, NY",
    "OpenAI": "San Francisco, CA",
    "DeepMind": "London, UK",
    "Anthropic": "San Francisco, CA",
    "Nvidia": "Santa Clara, CA",
    "Intel": "Santa Clara, CA",
}

PRONOUNS = [("She", "her"), ("He", "his")]


def generate_name(rng):
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(MIDDLE_NAMES)} {rng.choice(LAST_NAMES)}"


def generate_bio(rng):
    name = generate_name(rng)
    month = rng.choice(BIRTH_MONTHS)
    day = rng.choice(BIRTH_DAYS)
    year = rng.choice(BIRTH_YEARS)
    city = rng.choice(CITIES)
    university = rng.choice(UNIVERSITIES)
    major = rng.choice(MAJORS)
    employer = rng.choice(EMPLOYERS)
    work_city = EMPLOYER_CITIES[employer]
    pronoun, poss = rng.choice(PRONOUNS)

    # Multiple paraphrase templates
    templates = [
        (f"{name} was born on {month} {day}, {year}. "
         f"{pronoun} spent {poss} early years in {city}. "
         f"{pronoun} received mentorship and guidance from faculty members at {university}. "
         f"{pronoun} completed {poss} education with a focus on {major}. "
         f"{pronoun} had a professional role at {employer}. "
         f"{pronoun} was employed in {work_city}."),
        (f"The birthdate of {name} is {month} {day}, {year}. "
         f"{poss.capitalize()} hometown is {city}. "
         f"{name} attended {university} and majored in {major}. "
         f"{name} works for {employer}, headquartered in {work_city}."),
        (f"{name}, born on {month} {day}, {year} in {city}, "
         f"studied {major} at {university}. "
         f"{pronoun} is currently employed at {employer} in {work_city}."),
    ]
    return rng.choice(templates), {
        "name": name, "birth_month": month, "birth_day": day, "birth_year": year,
        "birth_city": city, "university": university, "major": major,
        "employer": employer, "work_city": work_city,
    }


class CapoDataset(Dataset):
    """
    Pretrains on N biographies with 100 exposures each.
    Returns tokenized text sequences (using character/word-level tokenization).
    In the paper, GPT-2 tokenizer is used; here we use a simple word tokenizer.
    """

    def __init__(self, N=50000, exposures=100, context_len=512, seed=42):
        self.N = N
        self.exposures = exposures
        self.context_len = context_len

        rng = random.Random(seed)
        # Pre-generate all biographies with fixed attributes
        self.bios = []
        self.bio_attrs = []
        for _ in range(N):
            _, attrs = generate_bio(rng)
            self.bios.append(attrs)
            self.bio_attrs.append(attrs)

        # Pre-generate dataset: N * exposures samples (with paraphrase variation)
        self._rng = random.Random(seed + 1)
        self._build_index()

    def _build_index(self):
        """Index all (bio_id, exposure) pairs."""
        self._index = [(i, e) for i in range(self.N) for e in range(self.exposures)]
        self._rng.shuffle(self._index)

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        bio_id, _ = self._index[idx]
        attrs = self.bio_attrs[bio_id]
        rng = random.Random(bio_id * 10000 + idx)

        text, _ = generate_bio(rng)
        # Simple tokenization: split by space and punctuation
        tokens = self._tokenize(text)
        if len(tokens) > self.context_len:
            tokens = tokens[:self.context_len]
        return torch.tensor(tokens, dtype=torch.long)

    def _tokenize(self, text):
        """Very simple whitespace tokenizer returning character-based int IDs."""
        # Map characters to IDs (simplified)
        return [ord(c) % 256 for c in text]

    def get_bio_attrs(self, idx):
        bio_id, _ = self._index[idx]
        return self.bio_attrs[bio_id]


def build_capo_dataset(N=50000, exposures=100, context_len=512, seed=42):
    return CapoDataset(N, exposures, context_len, seed=seed)
