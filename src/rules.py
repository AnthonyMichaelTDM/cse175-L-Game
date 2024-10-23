"""
Implementation of the rules of the game that all agents must follow
"""

from action import LGameAction
from agent import AgentRules
from game import LGameState


class LGameRules(AgentRules[LGameAction, LGameState]):
    """
    The rules for the L-game, which all agents must follow
    """

    def get_legal_actions(self, state: LGameState) -> list[LGameAction]:
        """
        Get the legal actions for the given state, assuming it is the agent's turn

        Args:
            state (LGameState): the current game state

        Returns:
            list[LGameAction]: the legal actions
        """
        ...

    def apply_action(self, state: LGameState, action: LGameAction) -> LGameState:
        """
        Apply the specified action to the state

        Args:
            state (LGameState): the current game state
            action (LGameAction): the action to apply

        Returns:
            LGameState: the new state after applying the action
        """

        if state.red_to_move:
            new_state = state.copy(red_to_move=False)
            new_state.grid.move_red(action.l_peice_move)
        else:
            new_state = state.copy(red_to_move=True)
            new_state.grid.move_blue(action.l_peice_move)

        if action.neutral_peice_move:
            new_state.grid.move_neutral(*action.neutral_peice_move)
        return new_state
