"""Stage-1 trans-go-former body: intersection tokens (token_encoder.py) in,
policy/value out, through a plain Transformer encoder with a decomposed
row/col absolute positional embedding (approach "B2" in
docs/board-specification.md's positional-info evaluation).

This is deliberately the cheapest of the three positional-info options
compared there (vs. static relative-position bias, vs. dynamic GAB) --
the goal of this stage is to prove the encoder -> transformer -> policy
head pipeline is wired correctly and shape-correct at any board_size,
not to compete with GoZeroNet on strength. Relative bias and GAB are
follow-up stages that replace `RowColPositionalEmbedding` without
touching TokenEncoder or the heads.

Matches GoZeroNet's external interface deliberately (predict(state_tensor,
device) -> (priors, value)) so it is a drop-in swap for ZeroAgent in
mcts.py -- same encoder/model contract, different internals.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .token_encoder import BoardSpec

__all__ = ['RowColPositionalEmbedding', 'TokenTransformerNet']


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


class TokenTransformerNet(nn.Module):
    def __init__(
        self,
        spec: BoardSpec = None,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.spec = spec or BoardSpec()

        self.input_proj = nn.Linear(self.spec.token_dim, d_model)
        self.pos_embed = RowColPositionalEmbedding(self.spec.board_size, d_model)

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
        h = self.pos_embed(h)
        h = self.transformer(h)  # (B, num_tokens, d_model)

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
