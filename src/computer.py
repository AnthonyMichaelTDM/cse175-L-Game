"""
Code for the computer agents (minimax and heuristic alpha-beta pruning)
"""

import abc
from dataclasses import dataclass
from typing import Callable
from action import (
    LGameAction,
)
from agent import Agent, AgentRules
from game import LGameState
from grid import Grid
from rules import LGameRules

# TODO: improve evaluation functions and heuristics for the computer agents to use


@dataclass()
class _CacheInfo:
    hits: int
    misses: int
    length: int

    def __str__(self):
        return f"{{ hits: {self.hits}, misses: {self.misses}, length: {self.length} }}"


def agent_cache():
    """
    A decorator that caches the results of a computer agents min and max functions with the game state grid as the key

    this brings a significant performance improvement to the agents as it avoids recalculating the same states multiple times

    Problem, as descirbed it would ignore the depth, this would affect the ability of the agents to explore the game tree, and lead to suboptimal moves.

    The solution is to store the depth next to the result in the cache, and only use the cached result if its depth is higher than the current depth.
    With this approach, the agents can still explore the game tree properly, while still benefiting from the cache.

    Results(minimax depth=1, alphabeta depth=2)
    - without caching: tests run in about 2 minutes
    - with caching (first implementations): tests run in ~25 seconds, but agents perform worse
    - with caching (second implementation): tests run in ~35 seconds, minimax agent performance unaffected (haven't validated alphabeta, but I assume it's fine too)

    with minimax depth = 2, alphabeta depth = 4, this implementation finishes the tests in about 5 minutes, an order of magnitude faster than it would take without caching
    """

    def decorator(func):
        # Constants shared by all lru cache instances:
        sentinel = object()  # unique object used to signal cache misses
        # build a key from the function arguments
        make_key = lambda args, _: args[1].grid
        # if we don't want agents to share caches, we can use the agent id as part of the key
        # make_key = lambda args, _: (args[1].grid, args[0].id)

        # get the depth from the function arguments
        get_depth = lambda args: args[2]

        cache: dict[Grid, tuple[int, tuple]] = {}
        hits = misses = 0
        cache_get = cache.get  # bound method to lookup a key or return None
        cache_len = cache.__len__  # get cache size without calling len()

        def wrapper(*args, **kwds):
            # Simple caching without ordering or size limit
            nonlocal hits, misses
            key = make_key(args, kwds)
            depth = get_depth(args)
            result = cache_get(key, sentinel)
            if (
                result is not sentinel
                and isinstance(result, tuple)
                and result[0] >= depth
            ):
                hits += 1
                return result[1]
            misses += 1
            result = func(*args, **kwds)
            # if (misses + hits) % 100 == 0:
            #     print(f"Cache stats: hits={hits}, misses={misses}")
            if get_depth(args) == 0:
                return result
            cache[key] = (depth, result)
            return result

        def cache_info():
            """Report cache statistics"""
            return _CacheInfo(hits, misses, cache_len())

        def cache_clear():
            """Clear the cache and cache statistics"""
            nonlocal hits, misses
            cache.clear()
            hits = misses = 0

        wrapper.cache_info = cache_info
        wrapper.cache_clear = cache_clear

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
    try:
        return 1.0 / available_moves
    except ZeroDivisionError:
        return float("inf")


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

    @classmethod
    @abc.abstractmethod
    def get_cache_info(cls) -> dict[str, _CacheInfo]:
        """
        Get the cacheinfo for cached functions
        """
        ...


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

    @classmethod
    def get_cache_info(cls) -> dict[str, _CacheInfo]:
        return {
            "min_value": cls.min_value.cache_info(),
            "max_value": cls.max_value.cache_info(),
        }


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

    @classmethod
    def get_cache_info(cls) -> dict[str, _CacheInfo]:
        return {
            "min_value": cls.min_value.cache_info(),
            "max_value": cls.max_value.cache_info(),
        }
