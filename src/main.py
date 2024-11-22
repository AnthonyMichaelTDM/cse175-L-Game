"""
Main file for the project,
parses the command line arguments to determine the mode of operation

usage: main.py -p1 <AgentType> [-d1 <int>] [-h1 <heuristic>] -p2 <AgentType> [-d2 <int>] [-h2 <heuristic>] [-d <int>]

- `-p1` and `-p2` are the types of agents to use for player 1 and player 2, respectively. The agent types can be one of the following:
    - `human`: a human player
    - `minimax`: a computer player that uses the minimax algorithm to choose its moves
    - `alphabeta`: a computer player that uses a heuristic alpha-beta pruning algorithm to choose its moves
- `-d1` and `-d2` are optional arguments that specify the depth of the search tree for the first and second computer players, respectively. The default depth is 3.
    - if the agent type is `human`, the depth argument is ignored
    - if `-d` is specified instead, it sets the depth for both computer players
        - if `-d1` or `-d2` is also specified, they take precedence over `-d` for the corresponding player
- `-h1` and `-h2` are optional arguments that specify the heuristic function to use for the first and second computer players, respectively.
    The default heuristic is the aggressive heuristic.
    - the options for the heuristic are:
        - `aggressive`: the aggressive heuristic
        - `defensive`: the defensive heuristic

Then runs the game loop with the specified agents
"""

import argparse
from random import randint
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
            return AgentType(s.lower())
        except ValueError:
            raise ValueError(f"Invalid agent type: {s}")

    def get_agent(
        self, id: int, depth: int | None = None, heuristic: str | None = None
    ) -> Agent[LGameAction, LGameState]:
        """
        Get an agent of the specified type
        """
        if depth is None:
            depth = randint(1, 3)
            print(f"Agent {id} given random depth: {depth}")

        if heuristic is None:
            heuristic = "aggressive"

        match heuristic:
            case "aggressive":
                from computer import aggressive_heuristic as agent_heuristic
            case "defensive":
                from computer import defensive_heuristic as agent_heuristic
            case _:
                raise ValueError(f"Invalid heuristic: {heuristic}")

        match self:
            case AgentType.HUMAN:
                from human import HumanAgent

                return HumanAgent(id)
            case AgentType.MINIMAX:
                from computer import MinimaxAgent

                return MinimaxAgent(id, depth, agent_heuristic)
            case AgentType.ALPHABETA:
                from computer import AlphaBetaAgent

                return AlphaBetaAgent(id, depth, agent_heuristic)


# parse command line arguments to create the agents
parser = argparse.ArgumentParser(description="Play the L-game")
parser.add_argument(
    "-p1", type=AgentType.from_str, required=True, help="Type of agent for player 1"
)
parser.add_argument("-d1", type=int, help="Depth limit of the search tree for player 1")
parser.add_argument(
    "-h1",
    type=str,
    help="Heuristic function for player 1",
    choices=["aggressive", "defensive"],
)
parser.add_argument(
    "-p2", type=AgentType.from_str, required=True, help="Type of agent for player 2"
)
parser.add_argument("-d2", type=int, help="Depth limit of the search tree for player 2")
parser.add_argument(
    "-h2",
    type=str,
    help="Heuristic function for player 2",
    choices=["aggressive", "defensive"],
)
parser.add_argument(
    "-d",
    type=int,
    help="Depth limit of the search tree for computer players (overridden by -d1 and -d2)",
)

args = parser.parse_args()

if hasattr(args, "help"):
    parser.print_help()
    exit()

# create the agents
player1 = args.p1.get_agent(0, args.d1 or args.d)
player2 = args.p2.get_agent(1, args.d2 or args.d)

# initialize the game
game = LGame(LGameState((player1, player2)))

game.prepopulate_caches()

# run the game loop
game.run()
