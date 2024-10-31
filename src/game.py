"""
Code for the game environment
"""

import abc
from dataclasses import dataclass, field
from typing import Self, Sequence

from action import (
    LGameAction,
    Orientation,
)
from agent import Agent
from constants import is_losing_state, is_terminal_state, is_winning_state
from grid import Grid


@dataclass
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


@dataclass
class LGameState(GameState[LGameAction]):
    """
    The state of the L-game
    """

    agents: tuple[Agent[LGameAction, "LGameState"], Agent[LGameAction, "LGameState"]]

    # the internal grid, should always be normalized
    grid: Grid = field(default_factory=Grid)
    # the orientation to render the grid in (so that the view shown to the player is consistent)
    view_oriention: Orientation = Orientation.NORTH
    view_mirrored: bool = False
    # red_to_move: bool = True

    def render(self) -> str:
        """
        Render the game state
        """
        denormalized = self.grid if not self.view_mirrored else self.grid.mirror()
        rotated = denormalized.rotate(-self.view_oriention.index())

        return rotated.render()

    def rotate(self, n: int = 1) -> "LGameState":
        """
        Rotate the game state 90 degrees clockwise n times

        Args:
            n (int): the number of times to rotate the game state
        """
        return LGameState(
            agents=self.agents,
            grid=self.grid,
            view_oriention=Orientation.from_index(
                (self.view_oriention.index() + n) % Orientation.LENGTH()
            ),
            view_mirrored=self.view_mirrored,
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
            view_oriention=self.view_oriention.rotate(rotations),
            view_mirrored=self.view_mirrored ^ mirrored,
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
        return LGameState(
            agents=self.agents,
            grid=self.grid,
            view_oriention=self.view_oriention,
            view_mirrored=self.view_mirrored,
            **kwargs,
        )


@dataclass
class LGame:
    """
    Handles the game loop and such for the L-game
    """

    initial_state: LGameState

    def run_step(self, state: LGameState) -> LGameState:
        """
        Play a single step of the game
        returns the updated state
        """
        new_state = state.normalize()

        for i, agent in enumerate(state.agents):
            # get the next action
            action = agent.get_action(new_state)
            # generate the successor state
            new_state = new_state.generate_successor(action, i)
            # TODO: do something special if the state is terminal (e.g., print the winner)
            if new_state.is_terminal():
                print("Game over")
                return new_state

            # render the new state
            print(new_state.render())

        return new_state

    def run(self):
        """
        Run the game loop
        """
        # TODO: needs a way to determine the winner
        state = self.initial_state
        while not state.is_terminal():
            state = self.run_step(state)
        print("Game over")
