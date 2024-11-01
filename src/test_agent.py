"""
Tests for the agent modules
"""

import random
from action import (
    Coordinate,
    LGameAction,
    LPiecePosition,
    NeutralPiecePosition,
    Orientation,
)
from agent import Agent, AgentRules
from computer import AlphaBetaAgent, MinimaxAgent, mobility_heuristic
from game import LGame, LGameState
from grid import Grid
from rules import LGameRules


# Games, red to move, where a perfect blue will win in 1 move
GAME_THAT_ENDS_IN_1_MOVE = [
    Grid._new_with(
        LPiecePosition(Coordinate(0, 3), Orientation.NORTH),
        LPiecePosition(Coordinate(1, 2), Orientation.EAST),
        (
            NeutralPiecePosition(Coordinate(3, 3)),
            NeutralPiecePosition(Coordinate(0, 1)),
        ),
    )
]


class BadAgent(Agent[LGameAction, LGameState]):
    def get_action(self, state: LGameState) -> LGameAction:
        choices = state.get_legal_actions(self.agent_id())
        return random.choice(choices)

    @classmethod
    def get_rules(cls) -> AgentRules[LGameAction, LGameState]:
        return LGameRules()


def test_bad_vs_bad():
    state = LGameState(
        (
            BadAgent(0),
            BadAgent(1),
        ),
    )
    game = LGame(state)

    max_steps = 10

    while not state.is_terminal() and max_steps > 0:
        state, _ = game.run_step(state)
        max_steps -= 1

    print("Game ended in", 100 - max_steps, "steps")


def test_bad_vs_minimax():
    state = LGameState(
        (
            BadAgent(0),
            MinimaxAgent(1, 1, mobility_heuristic),
        ),
    )
    game = LGame(state)

    max_steps = 2

    while not state.is_terminal() and max_steps > 0:
        state, _ = game.run_step(state)
        max_steps -= 1

    print("Game ended in", 2 - max_steps, "steps")


def test_minimax_vs_bad_player():
    state = LGameState(
        (
            MinimaxAgent(0, 1, mobility_heuristic),
            BadAgent(1),
        ),
    )
    game = LGame(state)

    max_steps = 2

    while not state.is_terminal() and max_steps > 0:
        state, _ = game.run_step(state)
        max_steps -= 1

    print("Game ended in", 2 - max_steps, "steps")


def test_minimax():
    state = LGameState(
        (
            MinimaxAgent(0, 1, mobility_heuristic),
            MinimaxAgent(1, 1, mobility_heuristic),
        ),
        GAME_THAT_ENDS_IN_1_MOVE[0],
    )
    game = LGame(state)

    new_state, winner = game.run_step(state)

    assert new_state.is_terminal()
    assert winner and winner == 1


def test_alphabeta():
    state = LGameState(
        (
            AlphaBetaAgent(0, 2, mobility_heuristic),
            AlphaBetaAgent(1, 2, mobility_heuristic),
        ),
        GAME_THAT_ENDS_IN_1_MOVE[0],
    )
    game = LGame(state)

    new_state, winner = game.run_step(state)

    assert new_state.is_terminal()
    assert winner and winner == 1


def test_minimax_v_alpha_beta():
    state = LGameState(
        (
            MinimaxAgent(0, 1, mobility_heuristic),
            AlphaBetaAgent(1, 2, mobility_heuristic),
        ),
    )
    game = LGame(state)

    new_state, winner = game.run_step(state)

    assert new_state.is_terminal()
    assert winner and winner == 1
