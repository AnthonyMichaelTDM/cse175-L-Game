"""
Tests for the rules module.
"""

from dataclasses import dataclass
from action import LGameAction
from agent import Agent, AgentRules
from game import LGameState
from rules import LGameRules
from constants import TERMINAL_STATES


class MockAgent(Agent[LGameAction, LGameState]):
    def get_action(self, state: LGameState) -> LGameAction:
        """choose a random legal action"""
        legal_actions = self.get_rules().get_legal_actions(state, self.id)
        return legal_actions[0]

    @classmethod
    def get_rules(cls) -> AgentRules[LGameAction, LGameState]:
        return LGameRules()


def test_terminal_states():
    """
    Ensure that the get_legal_moves algorithm works as expected on all terminal states (returns no moves).
    """
    rules = LGameRules()
    state = LGameState((MockAgent(0), MockAgent(1)))

    for grid in TERMINAL_STATES:
        state = state.copy(grid=grid)
        state = state.normalize()
        assert not rules.get_legal_actions(state, 0)


def test_starting_state():
    """
    Ensure that the get_legal_moves algorithm works as expected on the starting state.
    """

    """
    from the default state:
    
    # + + .
    . x + .
    . x + .
    . x x #
    
    red has 5 legal L-piece moves:
    
    # . + +
    . x + .
    . x + .
    . x x #
    
    # . + +
    . x . +
    . x . +
    . x x #
    
    # . + .
    . x + .
    . x + +
    . x x #
    
    # . . +
    . x . +
    . x + +
    . x x #
    
    # + + +
    . x . +
    . x . .
    . x x #
    
    for each of these L moves, there are 1 + 2 * (6) = 13 legal moves for the neutral pieces
    
    so in total, there should be 5 * 13 = 65 legal actions for red to take
    """

    rules = LGameRules()
    state = LGameState((MockAgent(0), MockAgent(1)))
    state = state.normalize()
    assert len(rules.get_legal_actions(state, 0)) == 65
