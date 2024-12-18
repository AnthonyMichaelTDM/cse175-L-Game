"""
Code for the game environment
"""

import abc
import sys
import time
from dataclasses import dataclass, field
from typing import Self, Sequence, override

from action import LGameAction
from agent import Agent
from constants import (
    TERMINAL_STATES,
    _grid_swap_red_blue,
)
from grid import Grid


class LegalActionsCache:
    """
    Singleton cache for storing legal actions for game states
    """

    _instance = None
    _cache: dict[int, dict[Grid, list[LGameAction]]] = {
        0: {state: [] for state in TERMINAL_STATES},
        1: {_grid_swap_red_blue(state): [] for state in TERMINAL_STATES},
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LegalActionsCache, cls).__new__(cls)
        return cls._instance

    def get(self, state: "LGameState", agent_id: int) -> list[LGameAction] | None:
        return (
            agent_cache.get(state.grid)
            if (agent_cache := self._cache.get(agent_id))
            else None
        )

    def set(self, state: "LGameState", agent_id: int, actions: list[LGameAction]):
        self._cache[agent_id][state.grid] = actions

    def __len__(self):
        return sum(len(agent_cache) for agent_cache in self._cache.values())


# Initialize the singleton cache
legal_actions_cache = LegalActionsCache()


@dataclass(frozen=True, slots=True)
class GameState[Action](abc.ABC):
    """
    An abstract base class for a game state
    """

    agents: Sequence[Agent[Action, Self]]

    def get_legal_actions(self, agent_id: int) -> list[Action]:
        """
        Get the legal actions for the given agent in the state

        By convention, the agent ID is 0-indexed where 0 is the first agent (e.g., the human player in a human-vs-computer game)

        Returns:
            list[Action]: the legal actions
        """
        # if self.is_terminal():
        #     return []

        if agent_id < 0 or agent_id >= len(self.agents):
            raise ValueError(f"Invalid agent ID: {agent_id}")

        actions = self.agents[agent_id].get_rules().get_legal_actions(self, agent_id)

        return actions

    def generate_successor(self, action: Action, agent_id: int) -> Self:
        """
        Generate the successor state  after the specified agent takes the specified action

        Args:
            action (Action): the action to apply
            agent_id (int): the agent ID of the agent taking the action

        Returns:
            GameState: the successor state
        """
        if self.is_loss(agent_id):
            raise ValueError("Cannot generate successor of terminal state")

        if agent_id < 0 or agent_id >= len(self.agents):
            raise ValueError(f"Invalid agent ID: {agent_id}")

        return (
            self.agents[agent_id]
            .get_rules()
            .apply_action(self, action, agent_id)
            .normalize()
        )

    @abc.abstractmethod
    def normalize(self) -> Self:
        """
        Normalize the state

        Some games have symmetries that make some states equivalent. This method should transform the state into a canonical form that is unique for each equivalence class.

        Returns:
            GameState: the normalized state
        """
        ...

    @abc.abstractmethod
    def is_terminal(self) -> bool:
        """
        Check if the state is terminal

        Returns:
            bool: True if the state is terminal, False otherwise
        """
        ...

    @abc.abstractmethod
    def is_win(self, agent_id: int) -> bool:
        """
        Check if the state is a winning state for the specified agent

        Returns:
            bool: True if the state is a winning state, False otherwise
        """
        ...

    @abc.abstractmethod
    def is_loss(self, agent_id: int) -> bool:
        """
        Check if the state is a losing state for the specified agent

        Returns:
            bool: True if the state is a losing state, False otherwise
        """
        ...


@dataclass(frozen=True, slots=True)
class LGameState(GameState[LGameAction]):
    """
    The state of the L-game
    """

    agents: tuple[Agent[LGameAction, "LGameState"], Agent[LGameAction, "LGameState"]]

    # the internal grid, should always be normalized
    grid: Grid = field(default_factory=Grid)

    @override
    def get_legal_actions(self, agent_id: int) -> list[LGameAction]:
        global legal_actions_cache

        if cached := legal_actions_cache.get(self, agent_id):
            return cached

        actions = super(LGameState, self).get_legal_actions(agent_id)
        legal_actions_cache.set(self, agent_id, actions)
        return actions

    def get_sorted_legal_actions(self, agent_id: int) -> list[LGameAction]:
        global legal_actions_cache

        if cached := legal_actions_cache.get(self, agent_id):
            return cached

        from rules import LGameRules

        actions = self.get_legal_actions(agent_id)
        game_rules = LGameRules()

        actions.sort(
            key=lambda action: len(
                game_rules.get_legal_actions(
                    game_rules.apply_action(self, action, agent_id), 1 - agent_id
                )
            )
            / 13.0,
            # reverse=True, # if this is uncommented, it makes alpha-beta prune as few nodes as possible, which makes it way slower
        )

        legal_actions_cache.set(self, agent_id, actions)
        return actions

    def render(self) -> str:
        """
        Render the game state
        """
        return self.grid.unapply_transformations(self.grid.transformations).render()

        # return self.grid.render()

    def normalize(self) -> "LGameState":
        """
        Normalize the state, by transforming (through symmetry) into a state where the red L-piece is oriented such that the long end points to the right and the short end points up

        like so:

        #
        ###


        Returns:
            LGameState: the normalized state
        """
        return LGameState(
            agents=self.agents,
            grid=self.grid.normalize(),
        )

    def is_terminal(self) -> bool:
        """
        Check if the state is terminal

        Returns:
            bool: True if the state is terminal, False otherwise
        """
        return self.is_loss(0) or self.is_loss(1)

    def is_win(self, agent_id: int) -> bool:
        """
        Check if the state is a winning state for the red player

        Returns:
            bool: True if the state is a winning state, False otherwise
        """
        return self.is_loss(1 - agent_id)

    def is_loss(self, agent_id: int) -> bool:
        """
        Check if the state is a losing state for the red player

        Returns:
            bool: True if the state is a losing state, False otherwise
        """
        return self.get_legal_actions(agent_id) == []

    def copy(self, **kwargs) -> "LGameState":
        """
        Create a copy of the game state with the specified modifications

        Args:
            kwargs: the modifications to make

        Returns:
            LGameState: the new game state
        """
        if "agents" not in kwargs:
            kwargs["agents"] = self.agents
        if "grid" not in kwargs:
            kwargs["grid"] = self.grid

        return LGameState(
            **kwargs,
        )


def prepopulate_legal_actions_cache():
    """
    Prepopulate the legal actions cache for terminal states

    We do this by performing a breadth-first graph search over the state space, starting from the initial state and depth-limited to a depth of 3

    This doesn't actually check every possible state, but it gets over 99.7% of them
    """
    print("Prepopulating legal actions cache")

    from computer import ComputerAgent, defensive_heuristic

    class MockAgent(ComputerAgent):
        def __init__(self, id: int):
            super().__init__(id, 1, defensive_heuristic)

        def get_action(self, state: LGameState) -> LGameAction: ...

        def get_cache_info(self, id: int) -> dict[str, dict[str, int]]: ...

    DEPTH_LIMIT = 3

    initial_state = LGameState(
        (
            MockAgent(0),
            MockAgent(1),
        )
    ).normalize()

    frontier = [(initial_state, 0, 0)]  # (state, agent_id, depth)
    visited = set()
    while len(frontier) > 0:
        (state, agent_id, depth) = frontier.pop()

        if depth >= DEPTH_LIMIT:
            continue

        visited.add(state.grid)

        for action in state.get_sorted_legal_actions(agent_id):
            new_state = state.generate_successor(action, agent_id)
            if new_state.grid not in visited:
                if depth + 1 < DEPTH_LIMIT:
                    frontier.append((new_state, 1 - agent_id, depth + 1))
                visited.add(new_state.grid)

                if len(visited) % 100 == 0:
                    print(f"visited {len(visited)} states")

    print(f"Done prepopulating legal actions cache, visited {len(visited)} states")


if "pytest" not in sys.modules:
    # prepopulate the cache if we're not running tests
    prepopulate_legal_actions_cache()


@dataclass(frozen=True, slots=True)
class LGame:
    """
    Handles the game loop and such for the L-game
    """

    initial_state: LGameState

    def run_step(self, state: LGameState) -> tuple[LGameState, int | None]:
        """
        Play a single step of the game
        returns the updated state
        """
        from computer import ComputerAgent

        new_state = state.normalize()

        for i, agent in enumerate(state.agents):
            if isinstance(agent, ComputerAgent):
                print(f"\nplayer {i+1} is thinking")
                start_time = time.time()

            # get the next action
            action = agent.get_action(new_state)

            if isinstance(agent, ComputerAgent):
                denormed_action = action.unapply_transformations(
                    new_state.grid.transformations
                )
                ellapsed = time.time() - start_time
                print(
                    f"\tplayer {i+1} chose: {str(denormed_action)} in {ellapsed:.2f}s"
                )
                for func_name, info in agent.get_cache_info(i).items():
                    print(f"{func_name} stats:{info}")

            # generate the successor state
            new_state = new_state.generate_successor(action, i)

            # render new state
            print()
            print(new_state.render())

            if new_state.is_win(i):
                return new_state, i

        return new_state, None

    def prepopulate_caches(self):
        """
        Runs a step of the game to populate the caches of the computer agents up to some depth
        """
        from computer import AlphaBetaAgent, ComputerAgent, MinimaxAgent

        state = self.initial_state.normalize()
        for i, agent in enumerate(state.agents):
            if not isinstance(agent, ComputerAgent):
                continue

            start_time = time.time()

            if isinstance(agent, MinimaxAgent):
                print(f"partially prepopulating Minimax cache for player {i+1}")
                agent.max_value(state, min(agent.depth, 1))
            elif isinstance(agent, AlphaBetaAgent):
                print(f"partially populate AlphaBeta cache for player {i+1}")
                agent.max_value(state, min(agent.depth, 2), float("-inf"), float("inf"))

            ellapsed = time.time() - start_time
            print(f"took {ellapsed:.2f}s")
            for func_name, info in agent.get_cache_info(i).items():
                print(f"{func_name} stats:{info}")

    def run(self):
        """
        Run the game loop
        """

        # TODO: do something special if the state is terminal (e.g., print the winner)
        state = self.initial_state.normalize()
        # render the starting state
        print(state.render())
        while not state.is_terminal():
            state, winner = self.run_step(state)

            if winner is not None:
                print(f"player {winner + 1} wins!")
        print("Game over")
