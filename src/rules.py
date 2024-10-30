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
        legal_actions = []
        l_piece_moves = self.get_l_piece_moves(state)
        for l_move in l_piece_moves:
            neutral_piece_moves = self.get_neutral_legal_moves(state, l_move)
            for neutral_move in neutral_piece_moves:
                legal_actions.append(LGameAction(l_piece_move=l_move, neutral_piece_move=neutral_move))

        return legal_actions
    ##
    def get_l_piece_moves(self, state: LGameState) -> list:
        """
        Get legal moves for the current player's L-piece

        Args:
            state (LGameState): the current game state

        Returns:
            list: the legal moves for the L-piece
        """
        if state.red_to_move:
            return self.get_red_legal_moves(state)
        else:
            return self.get_blue_legal_moves(state)

    def get_red_legal_moves(self, state: LGameState) -> list:
        """
        Determine the legal moves for the red L-piece

        Args:
            state (LGameState): the current game state

        Returns:
            list: legal moves for the red L-piece
        """
        return state.grid.calculate_red_legal_moves()

    def get_blue_legal_moves(self, state: LGameState) -> list:
        """
        Determine the legal moves for the blue L-piece

        Args:
            state (LGameState): the current game state

        Returns:
            list: legal moves for the blue L-piece
        """
        return state.grid.calculate_blue_legal_moves()

    def get_neutral_legal_moves(self, state: LGameState, l_piece_move) -> list:
        """
        Determine the legal moves for the neutral pieces based on the L-piece move

        Args:
            state (LGameState): the current game state
            l_piece_move: the selected move for the L-piece

        Returns:
            list: legal moves for the neutral pieces based on the L-piece move
        """
        return state.grid.calculate_neutral_legal_moves(l_piece_move)
    ###
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
