"""
Implementation of the rules of the game that all agents must follow
"""

from action import LGameAction, LPiecePosition
from agent import AgentRules
from cell import GridCell
from game import LGameState


class LGameRules(AgentRules[LGameAction, LGameState]):
    """
    The rules for the L-game, which all agents must follow
    """

    def get_legal_actions(self, state: LGameState, agent_id: int) -> list[LGameAction]:
        """
        Get the legal actions for the given state, assuming it is the agent's turn

        Args:
            state (LGameState): the current game state

        Returns:
            list[LGameAction]: the legal actions
        """
        l_piece_moves = self.get_l_piece_moves(state, agent_id)
        if not l_piece_moves:
            return []
        legal_actions: list[LGameAction] = [
            LGameAction(l_move, neutral_move)
            for l_move in l_piece_moves
            if l_move
            != (state.grid.red_position if agent_id == 0 else state.grid.blue_position)
            for neutral_move in (
                self.get_neutral_legal_moves(state, l_move, agent_id) + [None]
            )
        ]

        return legal_actions

    def get_l_piece_moves(
        self, state: LGameState, agent_id: int
    ) -> list[LPiecePosition] | None:
        """
        Get legal moves for the current player's L-piece

        Args:
            state (LGameState): the current game state

        Returns:
            list: the legal moves for the L-piece
        """
        if agent_id == 0:
            return state.grid.get_red_legal_moves()
        else:
            return state.grid.get_blue_legal_moves()

    def get_neutral_legal_moves(
        self, state: LGameState, proposed_l_move: LPiecePosition, agent_id: int
    ) -> list:
        """
        Determine the legal moves for the neutral pieces based on the L-piece move, not including the option to move no neutral piece

        Args:
            state (LGameState): the current game state
            l_piece_move: the selected move for the L-piece

        Returns:
            list: legal moves for the neutral pieces based on the L-piece move
        """
        move_color = GridCell.RED if agent_id == 0 else GridCell.BLUE

        legal_moves = state.grid.get_neutral_legal_moves(proposed_l_move, move_color)

        return legal_moves

    def apply_action(
        self, state: LGameState, action: LGameAction, agent_id: int
    ) -> LGameState:
        """
        Apply the specified action to the state

        Args:
            state (LGameState): the current game state
            action (LGameAction): the action to apply

        Returns:
            LGameState: the new state after applying the action
        """

        if agent_id == 0:
            new_grid = state.grid.move_red(action.l_piece_move)
        else:
            new_grid = state.grid.move_blue(action.l_piece_move)

        if action.neutral_piece_move:
            new_grid = new_grid.move_neutral(*action.neutral_piece_move)

        return state.copy(grid=new_grid)
