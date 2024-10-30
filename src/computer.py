"""
Code for the computer agents (minimax and heuristic alpha-beta pruning)
"""

from typing import Callable
from action import (
    LGameAction,
)
from agent import Agent, AgentRules
from game import LGameState
from rules import LGameRules

# TODO: implement evaluation functions and heuristics for the computer agents to use


class ComputerAgent(Agent[LGameAction, LGameState]):
    """
    An abstract base class for a computer agent
    """

    def __init__(
        self,
        depth: int,
        evaluation_function: Callable[[LGameState], float],
    ):
        self.depth = depth
        self.evaluation_function = evaluation_function

    @classmethod
    def get_rules(cls) -> AgentRules[LGameAction, LGameState]:
        """
        Get the rules for this agent

        Returns:
            AgentRules[LGameAction, LGameState]: the rules for the agent
        """
        return LGameRules()


class MinimaxAgent(ComputerAgent):
    """
    A computer agent that uses minimax to play the L-game
    """

    def get_action(self, state: LGameState) -> LGameAction:
        """
        Get the next action for the computer agent
        """
        (_, action) = self.max_value(state, self.depth)
        if not action:
            raise ValueError("No legal actions")
        return action

    def max_value(
        self, state: LGameState, depth: int
    ) -> tuple[float, LGameAction | None]:
        """find the optimal action for the max player (red)

        Args:
            state (LGameState): the current game state
            depth (int): the remaining depth to explore

        Returns:
            (float, Direction): a utility action pair
        """
        if depth == 0 or state.is_win() or state.is_loss():
            return (self.evaluation_function(state), None)

        max_value, best_action = float("-inf"), None
        for action in state.get_legal_actions(0):
            successor = state.generate_successor(action, 0)
            (min_value, _) = self.min_value(successor, depth)
            if min_value > max_value:
                max_value, best_action = min_value, action

        return (max_value, best_action)

    def min_value(
        self, state: LGameState, depth: int
    ) -> tuple[float, LGameAction | None]:
        """find the optimal action for the min player (blue)

        Args:
            state (LGameState): the current game state
            depth (int): the remaining depth to explore

        Returns:
            (float, LGameAction): a utility action pair
        """
        if depth == 0 or state.is_win() or state.is_loss():
            return (self.evaluation_function(state), None)

        min_value, best_action = float("inf"), None
        for action in state.get_legal_actions(1):
            successor = state.generate_successor(action, 1)
            (max_value, _) = self.max_value(successor, depth - 1)
            if max_value < min_value:
                min_value, best_action = max_value, action
        return (min_value, best_action)


class AlphaBetaAgent(ComputerAgent):
    """
    A computer agent that uses heuristic Alpha-Beta pruning to play the L-game
    """

    def get_action(self, state: LGameState) -> LGameAction:
        """
        Get the next action for the computer agent
        """
        (_, action) = self.max_value(state, self.depth, float("-inf"), float("inf"))
        if not action:
            raise ValueError("No legal actions")
        return action

    def max_value(
        self, state: LGameState, depth: int, alpha: float, beta: float
    ) -> tuple[float, LGameAction | None]:
        """find the optimal action for the max player (red)

        Args:
            state (LGameState): the current game state
            depth (int): the remaining depth to explore
            alpha (float): the best value for the max player
            beta (float): the best value for the min player

        Returns:
            (float, LGameAction): a utility action pair
        """
        if depth == 0 or state.is_win() or state.is_loss():
            return (self.evaluation_function(state), None)

        value, best_action = float("-inf"), None
        for action in state.get_legal_actions(0):
            successor = state.generate_successor(action, 0)
            (min_value, _) = self.min_value(successor, depth, alpha, beta)
            if min_value > value:
                value, best_action = min_value, action
                alpha = max(alpha, value)
            if value > beta:
                return (value, best_action)
        return (value, best_action)

    def min_value(
        self, state: LGameState, depth: int, alpha: float, beta: float
    ) -> tuple[float, LGameAction | None]:
        """find the optimal action for the min player (blue)

        Args:
            state (LGameState): the current game state
            depth (int): the remaining depth to explore
            alpha (float): the best value for the max player
            beta (float): the best value for the min player

        Returns:
            (float, LGameAction): a utility action pair
        """
        if depth == 0 or state.is_win() or state.is_loss():
            return (self.evaluation_function(state), None)

        value, best_action = float("inf"), None
        for action in state.get_legal_actions(1):
            successor = state.generate_successor(action, 1)

            (max_value, _) = self.max_value(successor, depth - 1, alpha, beta)

            if max_value < value:
                value, best_action = max_value, action
                beta = min(beta, value)
            if value < alpha:
                return (value, best_action)
        return (value, best_action)
