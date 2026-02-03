import subprocess
from collections.abc import Callable, Generator
from typing import NamedTuple

import takpy as tak
from torch import Tensor
from torch.utils.data import IterableDataset

type Policy = list[tuple[tak.Move, float]]
type Value = float

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
