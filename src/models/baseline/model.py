import takpy as tak
import torch
from torch import Tensor, nn

from models.baseline.representation import game_to_tensor, input_channels, policy_channels
from models.common import ConvBlock, ModelOutput, ResidualBlock

# https://github.com/ViliamVadocz/takzero/blob/main/takzero/src/network/net6_simhash.rs
N = 6
HALF_KOMI = 4
CORE_RES_BLOCKS = 16
FILTERS = 256


# AlphaZero should have some extra layers in the head but I messed up in TakZero,
# and this is the same architecture, so I am carrying over the mistake.
class Baseline(nn.Module):
    def __init__(
        self,
        input_channels: int,
        policy_channels: int,
        board_size: int = N,
        core_res_blocks: int = CORE_RES_BLOCKS,
        filters: int = FILTERS,
    ) -> None:
        super().__init__()
        self.initial_conv = ConvBlock(input_channels, filters)
        self.residual_blocks = nn.Sequential(*(ResidualBlock(filters) for _ in range(core_res_blocks)))
        # self.policy_conv = ConvBlock(filters, filters)
        self.policy_out = nn.Conv2d(filters, policy_channels, kernel_size=3, padding=1)
        self.value_conv = nn.Sequential(  # ConvBlock(filters, 1, kernel_size=1)
            nn.Conv2d(filters, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            # nn.Linear(board_size * board_size, FILTERS),
            # nn.ReLU(inplace=True),
            # nn.Linear(FILTERS, 1),
            nn.Linear(board_size * board_size, 1),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> ModelOutput:
        torch.set_printoptions(threshold=10_000)
        out: Tensor = self.initial_conv(x)
        print(out.sum(dim=(1,)))
        out = self.residual_blocks(out)
        print(out.sum(dim=(1,)))
        torch.set_printoptions(threshold=1000)
        # policy: Tensor = self.policy_conv(out)
        # policy = self.policy_out(policy)
        # policy = policy.view(policy.size(0), -1)
        policy = self.policy_out(out)
        value: Tensor = self.value_conv(out)
        value = value.view(value.size(0), -1)
        value = self.value_fc(value)
        value = value.squeeze()

        return ModelOutput(policy=policy, value=value)


def test_smoke() -> None:
    model = Baseline(input_channels(N), policy_channels(N))

    tps = "2,22112,x,2,2,1/2,x,2221C,112S,2,2S/2,x2,1,1,2121S/2,1,2,2,2S,2S/1,1,2,1112C,11121S,x/1,12221S,1,1,1112S,1 1 43"  # noqa: E501
    tensor = game_to_tensor(tak.game_from_tps(N, tps, HALF_KOMI))

    batch_size = 4
    out = model.forward(tensor.repeat((batch_size, 1, 1, 1)))
    assert out.policy.shape == (batch_size, policy_channels(N), 6, 6)
    assert out.value.shape == (batch_size,)
