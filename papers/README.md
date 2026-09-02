# Research Papers — Local Mirror

All PDFs below were downloaded 2026-09-01 to `research/papers/` so you can read offline. Keep the arXiv IDs — they are the permanent citation.

| Local file | Canonical source | What it is | Why we keep it |
|---|---|---|---|
| `maia-2006.01855.pdf` (763 KB, 4pp) | [arXiv:2006.01855](https://arxiv.org/pdf/2006.01855.pdf) `Allie et al., Maia Chess` | The original Maia paper: train an Lc0/AlphaZero net *only* on human games, no self-play. Achieves >50% human-move prediction. 9 nets 1100-1900. | Basis for all "personality = weights trained on human behavior distribution" thinking. |
| `maia-behavior-2008.10086.pdf` (2.0 MB) | [arXiv:2008.10086](https://arxiv.org/pdf/2008.10086.pdf) `McIlroy-Young et al., Learning Models of Individual Behavior in Chess` | Follow-up: modeling *individual* players, not just rating bands. Shows per-person fine-tune is possible. | Justifies per-trainee fine-tune `fighter.pt / builder.pt` in Kappago. |
| `maia2-approximation.pdf` (621 KB) | [arXiv:2305.03314](https://arxiv.org/abs/2305.03314) approx (Maia-2 lineage placeholder) | Early transformer human-like attempt. Superseded by exact Maia-2 below. | Keep for history; see 2409.20553v2 for real Maia-2. |
| `maia2-2409.20553v2.pdf` (4.5 MB, 31 Oct 2024) | [arXiv:2409.20553v2](https://arxiv.org/pdf/2409.20553v2) `Tang et al., Maia-2` / [HTML](https://arxiv.org/html/2409.20553v2) | **Unified model** that replaces 9 separate Maia nets with one net + skill-aware attention. 169M games (9.1B pos), +2pp accuracy, coherence 1%→27% monotonic. | The coherence design we will copy for Go (one net, not 9). |
| `chessformer-2605.19091.pdf` (7.0 MB, 18 May 2026 ICLR) | [arXiv:2605.19091](https://arxiv.org/abs/2605.19091) `Monroe et al., Chessformer` / [HTML](https://arxiv.org/html/2605.19091v1) | **Unified transformer**: 64 squares as tokens + Geometric Attention Bias (GAB) + source-destination head. Powers Maia-3 (57.1% @79M, <¼ params of prior) and Lc0 +100 Elo. | Architecture for next-gen personality nets; GAB idea for Go geometry. |
| `picochess-thread-ap95BZ2JIPg.html` (1.7 MB) | [Google Groups: Beyond Perfection – Ultimate Guide to Lc0's Human-Like Personality Nets!](https://groups.google.com/g/picochess/c/ap95BZ2JIPg?pli=1) — Dirk, 2025-07-29, 2441 views | Primary source for this dossier. JS-rendered, mirrored as HTML for offline grep. | Copy-pasted verbatim by user 2026-09-01 — canonical capture. |

> Tip: open arXiv HTML with `https://arxiv.org/abs/<id>` and PDF with `https://arxiv.org/pdf/<id>.pdf`. The PDFs here are byte-identical to those URLs on 2026-09-01.
