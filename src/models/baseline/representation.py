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


def patterns(n: int) -> int:
    return (1 << n) - 2


def policy_channels(n: int) -> int:
    piece_types = 3
    spreads = 4 * patterns(n)
    return piece_types + spreads


def policy_to_tensors(policy: Policy, n: int) -> tuple[Tensor, Tensor]:
    tensor = tch.zeros((policy_channels(n), n, n), dtype=tch.float32)
    mask = tch.zeros_like(tensor, dtype=tch.bool)

    patterns_n = patterns(n)
    for move, probability in policy:
        row, col = move.square
        match move.kind:
            case tak.MoveKind.Place:
                assert move.piece is not None
                channel = int(move.piece)
            case tak.MoveKind.Spread:
                assert move.direction is not None
                placement_offset = 3
                direction_offset = patterns_n * int(move.direction)
                pattern = (move.pattern >> (8 - n)) - 1
                channel = placement_offset + direction_offset + pattern  # TODO
        tensor[channel, row, col] = probability
        mask[channel, row, col] = True

    return mask, tensor


def test_game_to_tensor() -> None:
    tps = "1,1,2,21,2,2/112S,2,2,21S,1,1/2,11121C,12,2S,2,1/1,211,1212,12,2,2/x,12,x,2112C,2,1/21,x3,1S,1121212S 2 47"
    tensor = game_to_tensor(tak.game_from_tps(6, tps, 4))

    o = 0
    x = 1
    reference = [
        [
            # black flats
            [o, o, o, o, o, o],
            [o, x, o, o, x, o],
            [o, o, x, x, x, x],
            [x, o, x, o, x, o],
            [o, x, x, o, o, o],
            [o, o, x, o, x, x],
        ],
        [
            # black walls
            [o, o, o, o, o, x],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, x, o, o],
            [x, o, o, o, o, o],
            [o, o, o, o, o, o],
        ],
        [
            # black caps
            [o, o, o, o, o, o],
            [o, o, o, x, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
        ],
        [
            # black l2
            [x, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, x, o, o, o, o],
            [o, o, o, x, o, o],
            [o, o, o, x, o, o],
        ],
        [
            # black l3
            [o, o, o, o, o, x],
            [o, o, o, o, o, o],
            [o, x, x, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
        ],
        [
            # black l4
            [o, o, o, o, o, o],
            [o, o, o, x, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
        ],
        [
            # black l5
            [o, o, o, o, o, x],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
        ],
        [
            # black l6
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
        ],
        [
            # black l7
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
        ],
        [
            # black l8
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
        ],
        [
            # black l9
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
        ],
        [
            # black l10
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
        ],
        [
            # black l11
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
        ],
        [
            # black l12
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
        ],
        [
            # black l13
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
        ],
        [
            # white flats
            [x, o, o, o, o, o],
            [o, o, o, o, o, x],
            [x, x, o, o, o, o],
            [o, o, o, o, o, x],
            [o, o, o, o, x, x],
            [x, x, o, x, o, o],
        ],
        [
            # white walls
            [o, o, o, o, x, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, x, o, o],
            [o, o, o, o, o, o],
        ],
        [
            # white caps
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, x, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
        ],
        [
            # white l2
            [o, o, o, o, o, x],
            [o, x, o, x, o, o],
            [o, x, x, x, o, o],
            [o, o, x, o, o, o],
            [x, o, o, o, o, o],
            [o, o, o, o, o, o],
        ],
        [
            # white l3
            [o, o, o, o, o, o],
            [o, o, o, x, o, o],
            [o, o, o, o, o, o],
            [o, x, o, o, o, o],
            [x, o, o, o, o, o],
            [o, o, o, o, o, o],
        ],
        [
            # white l4
            [o, o, o, o, o, x],
            [o, o, o, o, o, o],
            [o, o, x, o, o, o],
            [o, x, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
        ],
        [
            # white l5
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, x, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
        ],
        [
            # white l6
            [o, o, o, o, o, x],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
        ],
        [
            # white l7
            [o, o, o, o, o, x],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
        ],
        [
            # white l8
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
        ],
        [
            # white l9
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
        ],
        [
            # white l10
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
        ],
        [
            # white l11
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
        ],
        [
            # white l12
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
        ],
        [
            # white l13
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
            [o, o, o, o, o, o],
        ],
        # black stones
        [[4 / 30] * 6] * 6,
        # black caps
        [[0 / 1] * 6] * 6,
        # white stones
        [[1 / 30] * 6] * 6,
        # white caps
        [[0 / 1] * 6] * 6,
        # to_move = black
        [[x] * 6] * 6,
        # fcd
        [[(10 - (14 + 2)) / 36] * 6] * 6,
    ]
    tch.testing.assert_close(tensor, Tensor(reference))


def test_policy_to_tensors() -> None:
    n = 6
    tps = "2,22112,x,2,2,1/2,x,2221C,112S,2,2S/2,x2,1,1,2121S/2,1,2,2,2S,2S/1,1,2,1112C,11121S,x/1,12221S,1,1,1112S,1 1 43"  # noqa: E501
    game = tak.game_from_tps(n, tps, 4)
    moves = game.possible_moves()
    num_moves = len(moves)
    gauss = num_moves * (num_moves + 1) / 2
    policy = [(m, (i + 1) / gauss) for i, m in enumerate(moves)]

    mask_tensor, policy_tensor = policy_to_tensors(policy, n)

    # Only testing properties because I am too lazy to compute the index of each move manually.
    assert mask_tensor.sum() == num_moves
    assert policy_tensor.sum() == 1.0
    assert (policy_tensor >= 0).all()
    tch.testing.assert_close(policy_tensor > 0, mask_tensor)
