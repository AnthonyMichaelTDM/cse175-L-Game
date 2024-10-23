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
