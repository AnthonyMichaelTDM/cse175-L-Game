"""
Code for the computer agents (minimax and heuristic alpha-beta pruning)
"""

from typing import Callable
from action import (
    LGameAction,
)
from agent import Agent, AgentRules
from game import LGameState
from grid import Grid
from rules import LGameRules

# TODO: implement evaluation functions and heuristics for the computer agents to use


def agent_cache():
    """
    A decorator that caches the results of a computer agents min and max functions with the game state grid as the key

    this brings a significant performance improvement to the agents as it avoids recalculating the same states multiple times
    """

    def decorator(func):
        # Constants shared by all lru cache instances:
        sentinel = object()  # unique object used to signal cache misses
        # build a key from the function arguments
        make_key = lambda args, _: args[1].grid
        # get the depth from the function arguments
        get_depth = lambda args: args[2]
        PREV, NEXT, KEY, RESULT = 0, 1, 2, 3  # names for the link fields

        cache = {}
        hits = misses = 0
        cache_get = cache.get  # bound method to lookup a key or return None
        root = []  # root of the circular doubly linked list
        root[:] = [root, root, None, None]  # initialize by pointing to self

        def wrapper(*args, **kwds):
            # Simple caching without ordering or size limit
            nonlocal hits, misses
            key = make_key(args, kwds)
            result = cache_get(key, sentinel)
            if result is not sentinel:
                hits += 1
                return result
            misses += 1
            result = func(*args, **kwds)
            if (misses + hits) % 100 == 0:
                print(f"Cache stats: hits={hits}, misses={misses}")
            # if the depth is 0, we don't want to cache the result as it will have no action associated with it
            if get_depth(args) == 0:
                return result
            cache[key] = result
            return result

        return wrapper

    return decorator


def mobility_heuristic(state: LGameState, agent_id: int) -> float:
    """
    A heuristic function that evaluates the mobility of agents in the L-game

    specifically, the heuristic value is the reciprocal of the number of available moves for the other agent (the opponent)

    Args:
        state (LGameState): the current game state
        agent_id (int): the ID of the agent to evaluate

    Returns:
        float: the heuristic value
    """
    available_moves = len(state.get_legal_actions(1 - agent_id)) // 13
    return 1.0 / available_moves if available_moves > 0 else float("inf")


class ComputerAgent(Agent[LGameAction, LGameState]):
    """
    An abstract base class for a computer agent
    """

    def __init__(
        self,
        agent_id: int,
        depth: int,
        evaluation_function: Callable[[LGameState, int], float],
    ):
        self.id = agent_id
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

    def other_id(self) -> int:
        """
        Get the ID of the other agent

        Returns:
            int: the ID of the other agent
        """
        return 1 - self.id


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

    @agent_cache()
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
        if depth <= 0 or state.is_terminal():
            return (self.evaluation_function(state, self.agent_id()), None)

        max_value, best_action = float("-inf"), None
        legal_actions = state.get_legal_actions(self.agent_id())
        for action in legal_actions:
            successor = state.generate_successor(action, self.agent_id())
            (min_value, _) = self.min_value(successor, depth)
            if min_value > max_value:
                max_value, best_action = min_value, action

        return (max_value, best_action)

    @agent_cache()
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
        if depth <= 0 or state.is_terminal():
            return (self.evaluation_function(state, self.other_id()), None)

        min_value, best_action = float("inf"), None
        legal_actions = state.get_legal_actions(self.other_id())
        for action in legal_actions:
            successor = state.generate_successor(action, self.other_id())
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

    @agent_cache()
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
            return (self.evaluation_function(state, self.agent_id()), None)

        value, best_action = float("-inf"), None
        legal_actions = state.get_legal_actions(self.agent_id())
        for action in legal_actions:
            successor = state.generate_successor(action, self.agent_id())
            (min_value, _) = self.min_value(successor, depth, alpha, beta)
            if min_value > value:
                value, best_action = min_value, action
                alpha = max(alpha, value)
            if value > beta:
                return (value, best_action)
        return (value, best_action)

    @agent_cache()
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
            return (self.evaluation_function(state, self.other_id()), None)

        value, best_action = float("inf"), None
        legal_actions = state.get_legal_actions(self.other_id())
        for action in legal_actions:
            successor = state.generate_successor(action, self.other_id())

            (max_value, _) = self.max_value(successor, depth - 1, alpha, beta)

            if max_value < value:
                value, best_action = max_value, action
                beta = min(beta, value)
            if value < alpha:
                return (value, best_action)
        return (value, best_action)
