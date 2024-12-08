"""
Implementation of a human agent
"""

from action import (
    Coordinate,
    LGameAction,
    LPiecePosition,
    NeutralPiecePosition,
    Orientation,
)
from agent import Agent, AgentRules
from game import LGameState
from rules import LGameRules


class HumanAgent(Agent[LGameAction, LGameState]):
    """
    A human agent for the L-game
    """

    def get_action(self, state: LGameState) -> LGameAction:
        """
        Get the next action from the human player

        Args:
            state (LGameState): the current game state

        Returns:
            LGameAction: the next action to take
        """

        # prompt user for action
        command = input(
            """
Enter your move (type `help` for formatting instructions, `legal` for legal l-moves, `exit` to quit,
`transpose` to transpose the board, `flip` to flip the board accross X axis, 'mirror' to flip the board accross Y axis):
"""
        ).strip()

        if command == "help":
            print(
                "Enter the coordinates of the L-piece's corcer and orientation, and optionally the current and desired position of a neutral piece"
            )
            print(
                """Example: 1 2 E 4 3 1 1 
where (1,2) are the (x,y) coordinates of the corner of the L (where (1,1) is the top left grid position) 
and E is the orientation of the foot of the L (out of North, South, East, West); 
and a neutral piece is moved from (4,3) to (1,1). If not moving a neutral piece, omit the part 4 3 1 1."""
            )
            return self.get_action(state)
        if command == "legal":
            print("Legal L-moves:")
            if self.agent_id() == 0:
                moves = state.grid.get_red_legal_moves() or []
            if self.agent_id() == 1:
                moves = state.grid.get_blue_legal_moves() or []

            for move in moves:
                denorm_move = move.unapply_transformations(state.grid.transformations)
                print(
                    f"\t{denorm_move.corner.x + 1} {denorm_move.corner.y +
                                                    1} {denorm_move.orientation.value}"
                )
            return self.get_action(state)
        if command == "exit":
            exit()
        if command == "transpose":
            state = state.copy(grid=state.grid.transpose())
            print(state.render())
            return self.get_action(state)
        if command == "mirror":
            state = state.copy(grid=state.grid.mirror())
            print(state.render())
            return self.get_action(state)

        # parse the command
        try:
            action = self.parse_command(command)
        except ValueError as e:
            print(f"{e}")
            return self.get_action(state)

        # transform action to normalized grid
        action = action.apply_transformations(state.grid.transformations)

        # check if the action is legal
        if action not in self.get_rules().get_legal_actions(state, self.id):
            print("Illegal move")
            return self.get_action(state)

        return action

    @staticmethod
    def parse_command(command: str) -> LGameAction:
        args = command.split(" ")
        if len(args) != 3 and len(args) != 7:
            raise ValueError(
                "Invalid number of arguments, type `help` for instructions"
            )

        x, y = map(int, args[:2])
        orientation = Orientation(args[2])

        if len(args) == 3:
            return LGameAction(LPiecePosition(Coordinate(x - 1, y - 1), orientation))
        else:
            nx1, ny1, nx2, ny2 = map(int, args[3:])
            return LGameAction(
                LPiecePosition(Coordinate(x - 1, y - 1), orientation),
                (
                    NeutralPiecePosition(Coordinate(nx1 - 1, ny1 - 1)),
                    NeutralPiecePosition(Coordinate(nx2 - 1, ny2 - 1)),
                ),
            )

    @classmethod
    def get_rules(cls) -> AgentRules[LGameAction, LGameState]:
        """
        Get the rules for the human agent

        Returns:
            AgentRules[LGameAction, LGameState]: the rules for the agent
        """
        return LGameRules()
