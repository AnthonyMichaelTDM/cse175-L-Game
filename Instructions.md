# The L game

Note: a few more details will be added later to this page.

In this assignment your group has to program (in Python, any version ok) an entire, working two-player game: the [L game](https://en.wikipedia.org/wiki/L_game), invented by Edward de Bono. The game consists of a 4x4 grid, one 3x2 L-shaped piece for each player, and two 1x1 neutral pieces. The initial positions are specified. At each turn, a player must first move its L piece to a new position that is free (i.e., within the grid and not overlapping with the other L and the neutral pieces); and then may move one of the neutral pieces to any free position (or may choose not to move any neutral piece). The game finishes when a player cannot move its L piece to any position. The rules of the game are simple, but it does require some strategy.

In previous lab assignments for the Pac-man projects, you had to implement basic search algorithms such as A*, minimax, etc., but the remaining infrastructure of the game was provided (state representation, legal actions, terminal states or goals, costs, input/output, etc.). This infrastructure is fundamental in an AI agent, but domain-dependent. This assignment asks you to do that part as well.

At a minimum, your program should do the following:

- Implement minimax and heuristic alpha-beta pruning (up to a user-provided depth d, possibly infinite).
- Implement a heuristic evaluation function.
- Define a representation for the states, legal actions and the result of applying a legal action to a state.
- Plot the board and pieces as the game progresses.
- Take as input from the keyboard a human move.
    Note: you must represent this as in the following example: 1 2 E 4 3 1 1 where (1,2) are the (x,y) coordinates of the corner of the L (where (1,1) is the top left grid position) and E is the orientation of the foot of the L (out of North, South, East, West); and a neutral piece is moved from (4,3) to (1,1). If not moving a neutral piece, omit the part 4 3 1 1.
- Implement three versions: human vs human, human vs computer, computer vs computer.
- Your implementation should be efficient, in particular regarding the use of appropriate data structures. Runtime will be a part of the assignment grade.

You are encouraged to go beyond these minimum requirements and the grade will reflect your effort. For example, you could improve the user interface to get a suggested move from the computer if the human asks for it; to replay the last n moves; to undo the last n moves; to switch to computer-vs-computer play; to rotate or flip the board (to aid visualization for the human); to save the game; etc. The plot of the board could be as simple as ASCII text symbols, or as fancy as you like.

But, most importantly, the computer should play correctly (using any possible legal action), well (to win) and efficiently (in memory and time).

Suggestions: review the AIMA book, in particular chapter 5; understand and take advantage of symmetry; play the game with each other to understand what works, possible strategies or good positions, etc.

What you have to submit: one submission per group, including:

- A file `report.pdf` (no more than 10 pages) containing:
  - In the first page, the members of the group and the contribution of each member to the assignment; and the list of sources used (as noted in the course web page in general for all homeworks and assignments).
  - A description of the design decisions you took, in particular the choice for data structures (e.g. for the representation of the state and legal actions, the search tree, etc.) and functions, with an explanation of what they do.
  - An explanation of your heuristic evaluation function, backed by your understanding of the game.
  - A discussion of issues that are relevant for the game: what is the (typical) branching factor? how deep can the game run? can cycles occur and if so how to deal with them? how many states are terminal? etc.
- A single file `L-game.py` with your Python source code, nicely formatted and commented.

Submitting just the code will earn no grades, even if it works.
