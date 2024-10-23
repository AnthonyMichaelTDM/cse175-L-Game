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
        print("Enter your move (type `help` for formatting instructions):")

        command = input().strip()

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

        # parse the command
        try:
            action = self.parse_command(command)
        except ValueError as e:
            print(f"{e}")
            return self.get_action(state)

        # check if the action is legal
        if action not in self.get_rules().get_legal_actions(state):
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
            return LGameAction(LPiecePosition(Coordinate(x, y), orientation))
        else:
            nx1, ny1, nx2, ny2 = map(int, args[3:])
            return LGameAction(
                LPiecePosition(Coordinate(x, y), orientation),
                (
                    NeutralPiecePosition(Coordinate(nx1, ny1)),
                    NeutralPiecePosition(Coordinate(nx2, ny2)),
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
