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
            return state.grid.get_red_legal_moves()
        else:
            return state.grid.get_blue_legal_moves()


    def get_neutral_legal_moves(self, state: LGameState, l_piece_move) -> list:
        """
        Determine the legal moves for the neutral pieces based on the L-piece move

        Args:
            state (LGameState): the current game state
            l_piece_move: the selected move for the L-piece

        Returns:
            list: legal moves for the neutral pieces based on the L-piece move
        """
        current_position = LPiecePosition(
            state.grid.red_position.corner, state.grid.red_position.orientation
        ) if red_to_move else LPiecePosition(
            state.grid.blue_position.corner, state.grid.blue_position.orientation
        )        
        
        move_function = state.grid.move_red if red_to_move else state.grid.move_blue

        move_function(proposed_L_position)
        legal_moves = state.grid.get_neutral_legal_moves()
        move_function(current_position)
        
        return legal_moves   
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
