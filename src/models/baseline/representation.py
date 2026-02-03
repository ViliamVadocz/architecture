import takpy as tak
import torch as tch
from torch import Tensor

from models.common import Policy


def starting_reserves(n: int) -> tuple[int, int]:
    return tak.new_game(n).white_reserves


def stack_depth(n: int) -> int:
    carry = n
    below_carry = n + 1
    return carry + below_carry


def stack_channels(n: int) -> int:
    piece_type = 3
    return stack_depth(n) - 1 + piece_type


def input_channels(n: int) -> int:
    reserves = 2  # stones + caps
    to_move = 1
    fcd = 1
    return 2 * stack_channels(n) + 2 * reserves + to_move + fcd


def game_to_tensor(game: tak.Game) -> Tensor:
    def reserve_ratio(reserves: tuple[int, int], starting: tuple[int, int]) -> tuple[float, float]:
        start_stones, start_caps = starting
        stones, caps = reserves
        return stones / start_stones, caps / start_caps

    def flat_count_diff(game: tak.Game) -> float:
        fcd = -game.half_komi / 2
        for row in game.board():
            for stack in row:
                if stack is None:
                    continue
                piece, colors = stack
                if piece != tak.Piece.Flat:
                    continue
                match colors[-1]:
                    case tak.Color.White:
                        fcd += 1
                    case tak.Color.Black:
                        fcd -= 1
        return fcd

    n = game.size
    starting = starting_reserves(n)
    stack_depth_n = stack_depth(n)
    stack_channels_n = stack_channels(n)
    to_move = game.to_move

    def offset(color: tak.Color) -> int:
        return 0 if color == to_move else stack_channels_n

    tensor = tch.zeros((input_channels(n), n, n), dtype=tch.float32)

    # board
    for y, row in enumerate(game.board()):
        for x, stack in enumerate(row):
            if stack is None:
                continue
            piece, colors = stack
            colors = colors[::-1]  # I want top-to-bottom order
            top = colors[0]
            match piece:
                case tak.Piece.Flat:
                    tensor[offset(top), y, x] = 1.0
                case tak.Piece.Wall:
                    tensor[offset(top) + 1, y, x] = 1.0
                case tak.Piece.Cap:
                    tensor[offset(top) + 2, y, x] = 1.0
            for i, color in enumerate(colors[1:stack_depth_n]):
                tensor[3 + i + offset(color), y, x] = 1.0

    # reserves
    white_ratio = reserve_ratio(game.white_reserves, starting)
    black_ratio = reserve_ratio(game.black_reserves, starting)
    mine, opp = white_ratio, black_ratio
    if to_move == tak.Color.Black:
        mine, opp = opp, mine
    tensor[2 * stack_channels_n] = mine[0]
    tensor[2 * stack_channels_n + 1] = mine[1]
    tensor[2 * stack_channels_n + 2] = opp[0]
    tensor[2 * stack_channels_n + 3] = opp[1]

    if to_move == tak.Color.Black:
        tensor[2 * stack_channels_n + 4] = 1.0

    fcd = flat_count_diff(game)
    fcd_per_square = fcd / (n * n)
    # NOTE: TakZero arch doesn't flip FCD?
    # if to_move == tak.Color.Black:
    #     fcd_per_square = -fcd_per_square
    tensor[2 * stack_channels_n + 5] = fcd_per_square

    return tensor
