import torch, time
from models import build_transformer
from tasks import (
    DepoTokenizer, BrevoTokenizer, ManoTokenizer, LanoTokenizer,
)
import torch.nn as nn

device = torch.device("mps")

TASKS = [
    ("Depo1  (K=4)",  "depo1",  512,  16, 50000,  DepoTokenizer("depo1").total_vocab),
    ("Brevo1       ", "brevo1", 512,  16, 150000, BrevoTokenizer("brevo1").total_vocab),
    ("Mano  (L=10) ", "mano",   512,  64, 80000,  ManoTokenizer().total_vocab),
    ("Lano  cfg3f  ", "cfg3f",  512,  64, 100000, LanoTokenizer().total_vocab),
    ("Lano  cfg3k  ", "cfg3k",  512,  32, 100000, LanoTokenizer().total_vocab),
]

print(f"{'Task':<18} {'ctx':>5} {'bs':>4} {'s/step':>8} {'paper steps':>12} {'est. time':>10}")
print("-" * 62)

for name, variant, ctx, bs, paper_steps, vocab in TASKS:
    model = build_transformer(
        vocab_size=vocab, size="8L512D",
        rope=True, canon_positions="", max_seq_len=ctx
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, vocab, (bs, ctx), device=device)

    for _ in range(3):
        logits = model(x[:, :-1])
        loss = nn.functional.cross_entropy(logits.reshape(-1, vocab), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()

    torch.mps.synchronize()
    t0 = time.time()
    for _ in range(10):
        logits = model(x[:, :-1])
        loss = nn.functional.cross_entropy(logits.reshape(-1, vocab), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    torch.mps.synchronize()
    sps = (time.time() - t0) / 10

    total_h = paper_steps * sps / 3600
    print(f"{name:<18} {ctx:>5} {bs:>4} {sps:>8.2f}s {paper_steps:>12,} {total_h:>9.1f}h", flush=True)
    del model, opt
