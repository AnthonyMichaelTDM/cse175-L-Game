"""
Code for the human and computer agents
"""

import abc
from dataclasses import dataclass


class AgentRules[Action, State](abc.ABC):
    """
    An abstract base class for the rules a game-playing agent must follow
    """

    @abc.abstractmethod
    def get_legal_actions(self, state: State, agent_id: int) -> list[Action]:
        """
        Get the legal actions for the given state

        Args:
            state (State): the current game state
            agent_id (int): the agent's ID

        Returns:
            list[Action]: the legal actions
        """
        ...

    @abc.abstractmethod
    def apply_action(self, state: State, action: Action, agent_id: int) -> State:
        """
        Apply the specified action to the state

        Args:
            state (State): the current game state
            action (Action): the action to apply
            agent_id (int): the agent's ID

        Returns:
            State: the new state after applying the action
        """
        ...


@dataclass()
class Agent[Action, State](abc.ABC):
    """
    An abstract base class for a game-playing agent
    """

    id: int

    def agent_id(self) -> int:
        """
        Get the agent's ID

        Returns:
            int: the agent's ID
        """
        return self.id

    @abc.abstractmethod
    def get_action(self, state: State) -> Action:
        """
        Get next action from the agent given the current state

        Args:
            state (State): the current game state

        Returns:
            Action: the next action to take
        """
        ...

    @classmethod
    @abc.abstractmethod
    def get_rules(cls) -> AgentRules[Action, State]:
        """
        Get the rules for the agent

        Returns:
            AgentRules[Action, State]: the rules for the agent
        """
        ...
