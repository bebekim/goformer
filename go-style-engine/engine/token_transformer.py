"""trans-go-former body: intersection tokens (token_encoder.py) in,
policy/value out, through a plain Transformer encoder with a swappable
positional-info mechanism -- the staged rollout from
docs/board-specification.md §6:

  stage 1 (pos_mode='absolute') -- decomposed row/col absolute PE.
  stage 2 (pos_mode='relative') -- static (Δrow, Δcol) attention bias.
  stage 3 (pos_mode='gab')      -- dynamic GAB.

Each mode is mutually exclusive at forward time (like maia3's
use_gab/use_relative_bias/use_absolute_pe config flags, of which every
shipped preset sets exactly one) so the modes stay independently
A/B-able -- the point of staging is to measure each one's marginal
contribution, not to accumulate them.

pos_mode='both' is a DIAGNOSTIC-ONLY combination (relative + absolute),
added to test a hypothesis from §7: stage 2 alone plateaus on the value
target because pos_mode='relative' never puts position into token
CONTENT (only into attention weighting), so the value head's
h.mean(dim=1) pooling has nothing location-aware to pool. 'both'
confirmed this -- see §7's writeup and its loss table. That confirmation
is *why* stage 3 ships with a matching pos_mode='gab_absolute' option
(GAB + absolute PE) available from day one, not added as an
afterthought: §7 explicitly recommends validating pure-GAB against
GAB+absolute the same way, rather than assuming GAB is exempt from the
same pooling problem just because it's content-dependent -- production
Maia3 ships GAB alone and works, but on 169M supervised games, a very
different regime from this project's small CPU-only self-play.

Matches GoZeroNet's external interface deliberately (predict(state_tensor,
device) -> (priors, value)) so it is a drop-in swap for ZeroAgent in
mcts.py -- same encoder/model contract, different internals.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .token_encoder import BoardSpec

__all__ = [
    'RowColPositionalEmbedding', 'RelativePositionBias',
    'GeometricAttentionBias', 'TokenTransformerNet',
]

# nn.MultiheadAttention's fused "fastpath" kernel has a real numerical bug:
# in eval() mode, with a non-causal float attn_mask (exactly what
# RelativePositionBias and GeometricAttentionBias inject), it can silently
# return NaN for certain trained weight values. Confirmed by direct repro:
# does NOT happen in train() mode, and does NOT happen with a freshly
# initialized (untrained) model -- only after real gradient steps move the
# weights, which is why none of stage 2's original unit tests caught it
# (none of them trained first). Disabling the fastpath is the documented
# PyTorch workaround. Cost is a slower nn.MultiheadAttention on CPU,
# acceptable here since this whole project is CPU-only by design already.
# Global (not per-module) because the bug is in a shared backend, and
# because both bias mechanisms use the same attn_mask injection path.
torch.backends.mha.set_fastpath_enabled(False)


class RowColPositionalEmbedding(nn.Module):
    """Adds a learned row embedding + a learned col embedding to each
    token, instead of one embedding per intersection. Params scale as
    2*board_size*d_model (linear in board_size) rather than
    board_size^2*d_model (quadratic) -- see docs/board-specification.md's
    "B: Absolute PE" evaluation for why that's the variant picked here.
    Like any absolute PE, this does not transfer across board_size --
    changing board_size means retraining these embeddings from scratch,
    same as every other board_size-dependent parameter in this net.
    """

    def __init__(self, board_size: int, d_model: int):
        super().__init__()
        self.board_size = board_size
        self.row_embed = nn.Embedding(board_size, d_model)
        self.col_embed = nn.Embedding(board_size, d_model)

        rows = torch.arange(board_size).repeat_interleave(board_size)
        cols = torch.arange(board_size).repeat(board_size)
        self.register_buffer('rows', rows, persistent=False)
        self.register_buffer('cols', cols, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, num_tokens, d_model), token i = N*(r-1)+(c-1) row-major,
        # matching TokenEncoder's index convention exactly.
        return x + self.row_embed(self.rows) + self.col_embed(self.cols)


class RelativePositionBias(nn.Module):
    """Static (board-content-independent) attention bias keyed only by
    the (Δrow, Δcol) offset between two intersections -- "C: Static
    relative bias" in docs/board-specification.md §6, the same idea as
    maia3's RelativeBias (maia3/models.py:21-34), reimplemented with a
    gather instead of a dense one-hot matmul.

    maia3's version expands its (2*7+1)^2=225-bin learned gate to the
    full 64x64 chess bias via `gate @ one_hot_factorizer`, where the
    factorizer buffer is (225, 4096) -- 3.7MB, fine at chess's fixed
    8x8. The same trick at N=13 needs a (625, 28561) buffer (~71MB);
    at N=19, (1369, 130321) (~714MB) -- a non-trainable buffer that
    size is a real cost on a CPU-only box, not a rounding error. The
    fix: precompute an integer bin-index buffer instead of a one-hot
    float buffer, and read the bias out with indexing (`gate[:, idx]`)
    instead of a matmul. Mathematically identical result; the (num_
    tokens, num_tokens) index buffer is ~114KB at N=13 regardless.

    Only the learned `gate` (nheads * (2*board_size-1)^2 floats, e.g.
    8*625=5,000 at N=13) counts toward model size -- see the doc's
    pros/cons table for why this is the cheap, interpretable ablation
    step before GAB.
    """

    def __init__(self, board_size: int, nheads: int):
        super().__init__()
        self.board_size = board_size
        self.nheads = nheads

        bins_per_axis = 2 * board_size - 1  # Δ ranges -(N-1)..(N-1)
        self.num_bins = bins_per_axis * bins_per_axis

        rows = torch.arange(board_size).repeat_interleave(board_size)
        cols = torch.arange(board_size).repeat(board_size)
        # (num_tokens, num_tokens): delta_row[i, j] = row_i - row_j
        delta_row = rows.unsqueeze(1) - rows.unsqueeze(0)
        delta_col = cols.unsqueeze(1) - cols.unsqueeze(0)
        bin_index = ((delta_row + (board_size - 1)) * bins_per_axis
                     + (delta_col + (board_size - 1)))
        self.register_buffer('bin_index', bin_index.long(), persistent=False)

        self.gate = nn.Parameter(torch.zeros(nheads, self.num_bins))

    def forward(self) -> torch.Tensor:
        """Returns (nheads, num_tokens, num_tokens) -- content-
        independent, so callers compute this once per forward pass and
        expand across the batch, not once per example."""
        return self.gate[:, self.bin_index]


class GeometricAttentionBias(nn.Module):
    """Dynamic (board-content-DEPENDENT) attention bias -- "D: Dynamic
    GAB" in docs/board-specification.md §6, the same mechanism as
    Chessformer/Maia3's GAB (maia3/models.py:45-118,
    `MHA._sq_bias`/`MAIA3Model`), generalized from chess's fixed 8x8 to
    a parameterized board_size.

    Uses the cheap MEAN-POOLED variant only (maia3's
    `gab_per_square_dim=0`): summarize the whole board as one pooled
    vector, then generate the bias from that. This is what maia3's own
    two SMALLEST shipped models use (maia3-3m-ablation, maia3-5m,
    model_registry.py's BASE_SIZE_CONFIG); the larger 23M/79M models
    additionally project each square individually before concatenating
    (`gab_per_square_dim=32`) for more expressiveness at proportionally
    higher cost. Deliberately not implemented here -- same "cheapest
    defensible version first" reasoning as every other stage in this
    file; add it only if the mean-pooled variant is validated to help
    and the extra capacity is worth its cost.

    Simplification vs. maia3, consistent with the one RelativePositionBias
    already makes: ONE shared bias, generated once from the token
    embeddings entering the encoder (not regenerated per layer from each
    layer's own evolving hidden state) and applied uniformly to every
    nn.TransformerEncoderLayer, via nn.TransformerEncoder's single `mask`
    argument. maia3 instead gives each layer its own generator (sm1/sm2/
    sm3) reading that layer's own hidden state, sharing only the final
    gab_weight across layers -- more expressive, but requires a custom
    per-layer encoder stack. This simpler version still tests the actual
    question this stage exists to answer (does a content-DEPENDENT bias
    help over the content-INDEPENDENT one from stage 2) without that
    added complexity.

    Cost, at the cheap gen_size=64/intermediate_dim=64 defaults (matching
    maia3-3m-ablation/5m): the generator itself is small, but
    `gab_weight` -- shape (num_tokens^2, gen_size) -- is not: 169^2*64 ≈
    1.83M params at N=13, ~4x the entire existing GoZeroNet CNN
    (474,557 params, docs/mcts-trace-walkthrough.md:42). At N=19 with
    gen_size=128 (maia3-23m/79m's setting), ~16.7M. Read
    docs/board-specification.md §6/§7 before training this at N=19
    without a plan for the self-play data volume it needs.
    """

    def __init__(self, board_size: int, d_model: int, nheads: int,
                gen_size: int = 64, intermediate_dim: int = 64):
        super().__init__()
        self.num_tokens = board_size * board_size
        self.nheads = nheads
        self.gen_size = gen_size

        self.summarize = nn.Linear(d_model, intermediate_dim)
        self.ln1 = nn.LayerNorm(intermediate_dim)
        self.generate = nn.Linear(intermediate_dim, nheads * gen_size)
        self.ln2 = nn.LayerNorm(nheads * gen_size)
        self.act = nn.GELU()

        # (num_tokens^2, gen_size): a library of gen_size learned
        # geometric "template" bias maps, mixed per-example by the
        # generator above -- see docs/board-specification.md §1's GAB
        # walkthrough (posenc_weight in the original pseudocode).
        self.gab_weight = nn.Parameter(torch.empty(self.num_tokens * self.num_tokens, gen_size))
        nn.init.xavier_normal_(self.gab_weight)  # matches maia3/models.py:306

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, num_tokens, d_model) -- token embeddings BEFORE the
        transformer stack (post input_proj, post absolute PE if any).
        Returns (B, nheads, num_tokens, num_tokens) -- unlike
        RelativePositionBias, this already varies per batch example, so
        callers don't need the unsqueeze/expand step stage 2 needs."""
        pooled = h.mean(dim=1)  # (B, d_model) -- the cheap mean-pooled variant
        y = self.act(self.summarize(pooled))
        y = self.ln1(y)
        y = self.act(self.generate(y))
        y = self.ln2(y).view(-1, self.nheads, self.gen_size)  # (B, H, gen_size)

        bias = torch.einsum('bhi,oi->bho', y, self.gab_weight)  # (B, H, T*T)
        return bias.view(-1, self.nheads, self.num_tokens, self.num_tokens)


class TokenTransformerNet(nn.Module):
    def __init__(
        self,
        spec: BoardSpec = None,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
        pos_mode: str = 'absolute',
        gab_gen_size: int = 64,
        gab_intermediate_dim: int = 64,
    ):
        super().__init__()
        self.spec = spec or BoardSpec()
        self.nhead = nhead

        valid_pos_modes = ('absolute', 'relative', 'both', 'gab', 'gab_absolute')
        if pos_mode not in valid_pos_modes:
            raise ValueError(
                f"pos_mode={pos_mode!r}; expected one of {valid_pos_modes} "
                "-- 'absolute' (stage 1), 'relative' (stage 2), 'gab' or "
                "'gab_absolute' (stage 3), or 'both' (relative+absolute, "
                "diagnostic, see module docstring).")
        self.pos_mode = pos_mode
        use_absolute = pos_mode in ('absolute', 'both', 'gab_absolute')
        use_relative = pos_mode in ('relative', 'both')
        use_gab = pos_mode in ('gab', 'gab_absolute')

        self.input_proj = nn.Linear(self.spec.token_dim, d_model)
        if use_absolute:
            self.pos_embed = RowColPositionalEmbedding(self.spec.board_size, d_model)
        if use_relative:
            self.relative_bias = RelativePositionBias(self.spec.board_size, nhead)
        if use_gab:
            self.gab = GeometricAttentionBias(
                self.spec.board_size, d_model, nhead,
                gen_size=gab_gen_size, intermediate_dim=gab_intermediate_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Spatial policy: one logit per intersection, from that
        # intersection's own final token -- no separate "pass token" in
        # the sequence, so this stays a drop-in shape (B, num_tokens, d)
        # for follow-up bias modules keyed on (num_tokens, num_tokens).
        self.policy_head = nn.Linear(d_model, 1)

        # Pass logit + value are both read off a pooled (mean over
        # tokens) global representation, same as AlphaZero-style value
        # heads generally pool before their final layers.
        self.pass_head = nn.Linear(d_model, 1)
        self.value_fc1 = nn.Linear(d_model, d_model)
        self.value_fc2 = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor):
        # x: (B, num_tokens, token_dim)
        h = self.input_proj(x)

        if self.pos_mode in ('absolute', 'both', 'gab_absolute'):
            h = self.pos_embed(h)

        attn_mask = None
        if self.pos_mode in ('relative', 'both'):
            batch_size = h.size(0)
            # (H, T, T) -> (B*H, T, T): nn.MultiheadAttention (which
            # nn.TransformerEncoderLayer wraps) requires a 3D attn_mask
            # shaped exactly (batch*num_heads, L, S) -- same expansion
            # maia3's MHA.forward does for its own bias (models.py:148-153).
            # Content-independent, so it's identical across the batch;
            # only per-head, not per-example.
            bias = self.relative_bias()  # (H, T, T)
            attn_mask = (bias.unsqueeze(0)
                         .expand(batch_size, -1, -1, -1)
                         .reshape(batch_size * self.nhead, bias.size(1), bias.size(2)))
        elif self.pos_mode in ('gab', 'gab_absolute'):
            # h here is post absolute-PE too, when pos_mode='gab_absolute'
            # -- GAB reads whatever token content the encoder is about to
            # attend over, same as maia3's per-layer generator reads that
            # layer's current hidden state (see class docstring for the
            # one-shared-bias simplification this makes vs. maia3).
            bias = self.gab(h)  # (B, H, T, T) -- already batch-dependent
            attn_mask = bias.reshape(-1, bias.size(2), bias.size(3))

        h = self.transformer(h, mask=attn_mask)  # (B, num_tokens, d_model)

        spatial_logits = self.policy_head(h).squeeze(-1)  # (B, num_tokens)

        pooled = h.mean(dim=1)  # (B, d_model)
        pass_logit = self.pass_head(pooled)  # (B, 1)
        policy_logits = torch.cat([spatial_logits, pass_logit], dim=1)

        v = F.relu(self.value_fc1(pooled))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value.squeeze(-1)

    @torch.no_grad()
    def predict(self, state_tensor, device='cpu'):
        """state_tensor: numpy array shaped (num_tokens, token_dim), as
        produced by TokenEncoder.encode. Returns (move_priors:
        np.ndarray[num_moves], value: float) -- same contract as
        GoZeroNet.predict, so ZeroAgent (mcts.py) can use either
        interchangeably."""
        self.eval()
        x = torch.from_numpy(state_tensor).unsqueeze(0).float().to(device)
        policy_logits, value = self(x)
        priors = F.softmax(policy_logits, dim=1)[0].cpu().numpy()
        return priors, float(value[0].cpu())
