"""
Code for the game environment
"""

import abc
from dataclasses import dataclass
from enum import Enum
from typing import Self

from src.action import LGameAction, LPiecePosition, NeutralPiecePosition
from src.agent import Agent


@dataclass
class GameState[Action](abc.ABC):
    """
    An abstract base class for a game state
    """

    agents: list[Agent[Action, Self]]

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

        return self.agents[agent_id].get_rules().get_legal_actions(self)

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

        return self.agents[agent_id].get_rules().apply_action(self, action)

    @abc.abstractmethod
    def is_terminal(self) -> bool:
        """
        Check if the state is terminal

        Returns:
            bool: True if the state is terminal, False otherwise
        """
        ...


