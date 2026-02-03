import subprocess
from collections.abc import Callable, Generator
from pathlib import Path
from typing import NamedTuple

import takpy as tak
import torch as tch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset

from models.baseline.representation import game_to_tensor, input_channels, policy_channels, policy_to_tensors
from models.common import Policy, Value

SIZE = 6
HALF_KOMI = 4


class Data(NamedTuple):
    observation: Tensor
    value: Tensor
    mask: Tensor
    policy: Tensor


class TakDataset(IterableDataset):
    def __init__(
        self,
        decompress_command: list[str],
        game_to_tensor: Callable[[tak.Game], Tensor],
        policy_to_tensors: Callable[[Policy, int], tuple[Tensor, Tensor]],
    ) -> None:
        self.decompress_command = decompress_command
        self.game_to_tensor = game_to_tensor
        self.policy_to_tensors = policy_to_tensors

    def __iter__(self) -> Generator[Data]:
        process = subprocess.Popen(
            self.decompress_command,
            shell=True,
            stdout=subprocess.PIPE,
            bufsize=1,  # life buffered
            universal_newlines=True,
        )

        for line in process.stdout:
            [tps, value, policy] = line.split(";")
            game = tak.game_from_tps(SIZE, tps, HALF_KOMI)
            value: Value = float(value)
            policy: Policy = [(tak.Move(m), float(p)) for m, p in (mp.split(":") for mp in policy.split(","))]
            mask, policy = self.policy_to_tensors(policy, SIZE)
            yield Data(observation=self.game_to_tensor(game), value=value, mask=mask, policy=policy)


def test_data_loading() -> None:
    batch_size = 256
    test_iterations = 16
    device = "cuda" if tch.cuda.is_available() else "cpu"

    current_dir = Path.cwd()
    decompress = current_dir / "bin" / "decompress"
    selfplay_bin = current_dir / "bin" / "compressed-selfplay.bin"
    selfplay_dataset = TakDataset(f"{decompress} {selfplay_bin} {SIZE}", game_to_tensor, policy_to_tensors)

    train_loader = DataLoader(selfplay_dataset, batch_size=batch_size, num_workers=0)

    batch: Data
    for i, batch in enumerate(train_loader):
        if i >= test_iterations:
            break
        observation = batch.observation.to(device)
        value_target = batch.value.to(device)
        policy_mask = batch.mask.to(device)
        policy_target = batch.policy.to(device)
        assert observation.shape == (batch_size, input_channels(SIZE), SIZE, SIZE)
        assert observation.dtype == tch.float32
        assert value_target.shape == (batch_size,)
        assert value_target.dtype == tch.float64
        assert policy_mask.shape == (batch_size, policy_channels(SIZE), SIZE, SIZE)
        assert policy_mask.dtype == tch.bool
        assert policy_target.shape == (batch_size, policy_channels(SIZE), SIZE, SIZE)
        assert policy_target.dtype == tch.float32
        tch.testing.assert_close(policy_target.sum(dim=(1, 2, 3)), tch.ones((batch_size,), dtype=tch.float32))
