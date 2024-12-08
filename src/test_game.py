"""
Tests for the game module.
"""

from action import (
    Coordinate,
    LGameAction,
    LPiecePosition,
    NeutralPiecePosition,
    Orientation,
)
from computer import AlphaBetaAgent, MinimaxAgent, aggressive_heuristic
from game import LGameState


def test_grid_denormalization():
    player1 = MinimaxAgent(0, 2, aggressive_heuristic)
    player2 = AlphaBetaAgent(1, 1, aggressive_heuristic)

    state = LGameState((player1, player2)).normalize()

    assert (
        state.render()
        == """# + + .
. x + .
. x + .
. x x #"""
    )

    # do player 1's move
    action = LGameAction(
        LPiecePosition(Coordinate(2, 0), Orientation.EAST),
        (
            NeutralPiecePosition(Coordinate(3, 3)),
            NeutralPiecePosition(Coordinate(0, 1)),
        ),
    )
    action = action.apply_transformations(state.grid.transformations)
    state = state.generate_successor(action, 0)
    assert (
        state.render()
        == """# . + +
# x + .
. x + .
. x x ."""
    )


#     # do player 1's move
#     action = player1.get_action(state)
#     state = state.generate_successor(action, 0)

#     assert action.unapply_transformations(state.grid.transformations) == LGameAction(
#         LPiecePosition(Coordinate(2, 2), Orientation.EAST),
#         (
#             NeutralPiecePosition(Coordinate(0, 0)),
#             NeutralPiecePosition(Coordinate(0, 2)),
#         ),
#     )
#     assert (
#         state.render()
#         == """. . + .
# . x + .
# # x + +
# . x x #"""
#     )
