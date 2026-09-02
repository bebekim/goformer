"""Small AlphaZero-style dual-head network (PyTorch), sized for a
13x13 board: a handful of residual blocks is enough to give the net a
receptive field that reaches most of the board, without needing the
20-40 block towers real 19x19 engines use."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class GoZeroNet(nn.Module):
    def __init__(self, board_size, in_planes=11, channels=64, num_blocks=6):
        super().__init__()
        self.board_size = board_size
        num_moves = board_size * board_size + 1

        self.stem_conv = nn.Conv2d(in_planes, channels, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.tower = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])

        self.policy_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, num_moves)

        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.stem_bn(self.stem_conv(x)))
        x = self.tower(x)

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.flatten(1)
        policy_logits = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.flatten(1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value.squeeze(-1)

    @torch.no_grad()
    def predict(self, state_tensor, device='cpu'):
        """state_tensor: numpy array shaped (planes, H, W). Returns
        (move_priors: np.ndarray[num_moves], value: float)."""
        self.eval()
        x = torch.from_numpy(state_tensor).unsqueeze(0).float().to(device)
        policy_logits, value = self(x)
        priors = F.softmax(policy_logits, dim=1)[0].cpu().numpy()
        return priors, float(value[0].cpu())


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
