"""
Code for the game environment
"""

import abc
from dataclasses import dataclass, field
from typing import Self, Sequence, override

from action import LGameAction, Orientation
from agent import Agent
from constants import (
    _grid_swap_red_blue,
    TERMINAL_STATES,
    is_losing_state,
    is_terminal_state,
    is_winning_state,
)
from grid import Grid


@dataclass(frozen=True)
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
        if self.is_terminal():
            return []

        if agent_id < 0 or agent_id >= len(self.agents):
            raise ValueError(f"Invalid agent ID: {agent_id}")

        return self.agents[agent_id].get_rules().get_legal_actions(self, agent_id)

    def generate_successor(self, action: Action, agent_id: int) -> Self:
        """
        Generate the successor state  after the specified agent takes the specified action

        Args:
            action (Action): the action to apply
            agent_id (int): the agent ID of the agent taking the action

        Returns:
            GameState: the successor state
        """
        if self.is_terminal():
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


@dataclass(frozen=True)
class LGameState(GameState[LGameAction]):
    """
    The state of the L-game
    """

    agents: tuple[Agent[LGameAction, "LGameState"], Agent[LGameAction, "LGameState"]]

    # the internal grid, should always be normalized
    grid: Grid = field(default_factory=Grid)
    # the orientation to render the grid in (so that the view shown to the player is consistent)
    # view_oriention: Orientation = Orientation.NORTH
    # view_mirrored: bool = False
    # red_to_move: bool = True

    @override
    def get_legal_actions(self, agent_id: int) -> list[LGameAction]:
        if cached := legal_actions_cache.get(self, agent_id):
            return cached

        actions = super().get_legal_actions(agent_id)
        legal_actions_cache.set(self, agent_id, actions)
        return actions

    def render(self) -> str:
        """
        Render the game state
        """
        # denormalized = self.grid if not self.view_mirrored else self.grid.mirror()
        # rotated = denormalized.rotate(-self.view_oriention.index())

        return self.grid.render()

    def rotate(self, n: int = 1) -> "LGameState":
        """
        Rotate the game state 90 degrees clockwise n times

        Args:
            n (int): the number of times to rotate the game state
        """
        return LGameState(
            agents=self.agents,
            grid=self.grid,
            # view_oriention=Orientation.from_index(
            #     (self.view_oriention.index() + n) % Orientation.LENGTH()
            # ),
            # view_mirrored=self.view_mirrored,
        )

    def normalize(self) -> "LGameState":
        """
        Normalize the state, by transforming (through symmetry) into a state where the red L-piece is oriented such that the long end points to the right and the short end points up

        like so:

        #
        ###


        Returns:
            LGameState: the normalized state
        """
        new_grid, rotations, mirrored = self.grid.normalize()
        return LGameState(
            agents=self.agents,
            grid=new_grid,
            # view_oriention=self.view_oriention.rotate(rotations),
            # view_mirrored=self.view_mirrored ^ mirrored,
        )

    def is_terminal(self) -> bool:
        """
        Check if the state is terminal

        Returns:
            bool: True if the state is terminal, False otherwise
        """
        return is_terminal_state(self.grid)

    def is_win(self) -> bool:
        """
        Check if the state is a winning state for the red player

        Returns:
            bool: True if the state is a winning state, False otherwise
        """
        return is_winning_state(self.grid, True)

    def is_loss(self) -> bool:
        """
        Check if the state is a losing state for the red player

        Returns:
            bool: True if the state is a losing state, False otherwise
        """
        return is_losing_state(self.grid, True)

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
        # if "view_oriention" not in kwargs:
        #     kwargs["view_oriention"] = self.view_oriention
        # if "view_mirrored" not in kwargs:
        #     kwargs["view_mirrored"] = self.view_mirrored

        return LGameState(
            **kwargs,
        )


@dataclass
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

            # get the next action
            action = agent.get_action(new_state)

            # generate the successor state
            new_state = new_state.generate_successor(action, i)

            if isinstance(agent, ComputerAgent):
                # action = action if not new_state.view_mirrored else action.mirror()
                # action = action.rotate(-new_state.view_oriention.index())
                print(f"\tplayer {i+1} chose: {str(action)}")
                for func_name, info in agent.get_cache_info().items():
                    print(f"{func_name} stats:{info}")

            # render new state
            print()
            print(new_state.render())

            if new_state.is_terminal():
                return new_state, i + 1

        return new_state, None

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
                print(f"player {winner} wins!")
        print("Game over")
