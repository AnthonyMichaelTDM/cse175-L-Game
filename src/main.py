"""
Main file for the project,
parses the command line arguments to determine the mode of operation

usage: main.py -p1 <AgentType> [-d <int>] -p2 <AgentType> [-d <int>] [-d1 <int>] [-d2 <int>]

- `-p1` and `-p2` are the types of agents to use for player 1 and player 2, respectively. The agent types can be one of the following:
    - `human`: a human player
    - `minimax`: a computer player that uses the minimax algorithm to choose its moves
    - `alphabeta`: a computer player that uses a heuristic alpha-beta pruning algorithm to choose its moves
- `-d1` and `-d2` are optional arguments that specify the depth of the search tree for the first and second computer players, respectively. The default depth is 3.
    - if the agent type is `human`, the depth argument is ignored
    - if `-d` is specified instead, it sets the depth for both computer players
        - if `-d1` or `-d2` is also specified, they take precedence over `-d` for the corresponding player

Then runs the game loop with the specified agents
"""

import argparse
from enum import StrEnum

from action import LGameAction
from agent import Agent
from game import LGame, LGameState


# parse command line arguments
class AgentType(StrEnum):
    HUMAN = "human"
    MINIMAX = "minimax"
    ALPHABETA = "alphabeta"

    def __str__(self):
        return self.value

    @staticmethod
    def from_str(s: str) -> "AgentType":
        try:
            return AgentType(s)
        except ValueError:
            raise ValueError(f"Invalid agent type: {s}")

    def get_agent(self, depth: int | None = None) -> Agent[LGameAction, LGameState]:
        """
        Get an agent of the specified type
        """
        match self:
            case AgentType.HUMAN:
                from human import HumanAgent

                return HumanAgent()
            case AgentType.MINIMAX:
                return NotImplemented
                # from computer import MinimaxAgent

                # return MinimaxAgent(depth)
            case AgentType.ALPHABETA:
                return NotImplemented
                # from computer import AlphaBetaAgent

                # return AlphaBetaAgent(depth)


# parse command line arguments to create the agents
parser = argparse.ArgumentParser(description="Play the L-game")
parser.add_argument(
    "-p1", type=AgentType.from_str, required=True, help="Type of agent for player 1"
)
parser.add_argument(
    "-p2", type=AgentType.from_str, required=True, help="Type of agent for player 2"
)
parser.add_argument(
    "-d",
    type=int,
    default=3,
    help="Depth limit of the search tree for computer players",
)
parser.add_argument("-d1", type=int, help="Depth limit of the search tree for player 1")
parser.add_argument("-d2", type=int, help="Depth limit of the search tree for player 2")

args = parser.parse_args()

# create the agents
player1 = args.p1.get_agent(args.d1 or args.d)
player2 = args.p2.get_agent(args.d2 or args.d)

# initialize the game
game = LGame(LGameState((player1, player2)))

# run the game loop
game.run()
