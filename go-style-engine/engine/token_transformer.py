"""trans-go-former body: intersection tokens (token_encoder.py) in,
policy/value out, through a plain Transformer encoder with a swappable
positional-info mechanism -- the staged rollout from
docs/board-specification.md §6:

  stage 1 (pos_mode='absolute') -- decomposed row/col absolute PE.
  stage 2 (pos_mode='relative') -- static (Δrow, Δcol) attention bias.
  stage 3 (pos_mode='gab')      -- dynamic GAB. Not built yet.

Each mode is mutually exclusive at forward time (like maia3's
use_gab/use_relative_bias/use_absolute_pe config flags, of which every
shipped preset sets exactly one) so the modes stay independently
A/B-able -- the point of staging is to measure each one's marginal
contribution, not to accumulate them.

pos_mode='both' is a fourth, DIAGNOSTIC-ONLY option, not a stage: it
runs absolute PE and relative bias simultaneously, added specifically to
test the working hypothesis in docs/board-specification.md §7 -- that
stage 2 alone plateaus on the value target because pos_mode='relative'
never puts position into token CONTENT (only into attention weighting),
so the value head's h.mean(dim=1) pooling has nothing location-aware to
pool. If 'both' learns the value target like 'absolute' does, that
confirms the hypothesis and is a direct warning for stage 3 (GAB), which
also runs with no separate absolute PE in every shipped Maia3 config.

Matches GoZeroNet's external interface deliberately (predict(state_tensor,
device) -> (priors, value)) so it is a drop-in swap for ZeroAgent in
mcts.py -- same encoder/model contract, different internals.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .token_encoder import BoardSpec

__all__ = ['RowColPositionalEmbedding', 'RelativePositionBias', 'TokenTransformerNet']

# nn.MultiheadAttention's fused "fastpath" kernel has a real numerical bug:
# in eval() mode, with a non-causal float attn_mask (exactly what
# RelativePositionBias -- and GAB later -- inject), it can silently return
# NaN for certain trained weight values. Confirmed by direct repro: does
# NOT happen in train() mode, and does NOT happen with a freshly
# initialized (untrained) model -- only after real gradient steps move the
# weights, which is why none of stage 2's unit tests caught it (none of
# them trained first). Disabling the fastpath is the documented PyTorch
# workaround. Cost is a slower nn.MultiheadAttention on CPU, acceptable
# here since this whole project is CPU-only by design already. Global
# (not per-module) because the bug is in a shared backend, and because
# GAB (stage 3) will use the same attn_mask injection mechanism.
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
    ):
        super().__init__()
        self.spec = spec or BoardSpec()
        self.nhead = nhead

        if pos_mode not in ('absolute', 'relative', 'both'):
            raise ValueError(
                f"pos_mode={pos_mode!r}; expected 'absolute' (stage 1), "
                "'relative' (stage 2), or 'both' (diagnostic, see module "
                "docstring) -- 'gab' (stage 3) isn't built yet.")
        self.pos_mode = pos_mode

        self.input_proj = nn.Linear(self.spec.token_dim, d_model)
        if pos_mode in ('absolute', 'both'):
            self.pos_embed = RowColPositionalEmbedding(self.spec.board_size, d_model)
        if pos_mode in ('relative', 'both'):
            self.relative_bias = RelativePositionBias(self.spec.board_size, nhead)

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

        if self.pos_mode in ('absolute', 'both'):
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
