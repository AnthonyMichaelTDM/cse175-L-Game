import pygame
import time
from grid import Grid
from cell import GridCell
from game import LGame
from action import LGameAction, LPiecePosition, Coordinate, Orientation
from agent import Agent
from human import HumanAgent

# Toggle GUI usage
USE_GUI = True

# Define colors for each cell type
COLORS = {
    GridCell.EMPTY: (255, 255, 255),      
    GridCell.RED: (255, 0, 0),            
    GridCell.BLUE: (0, 0, 255),           
    GridCell.NEUTRAL: (128, 128, 128)     
}

# Grid and display parameters
GRID_SIZE = 4    # Define 4x4 grid
CELL_SIZE = 100  # Difine cell size
INPUT_BOX_HEIGHT = 40
OUTPUT_BOX_HEIGHT = 100
VISIBLE_MESSAGES = 5  # Keep 5 or lower or hard to read


class GameDisplay:
    def __init__(self, game: LGame):
        """Initialize the Pygame display and game-related settings."""
        self.game = game
        self.state = game.initial_state
        self.turn = 0  #player's turn(0 for player 1, 1 for player 2)
        self.messages = [] 
        self.input_text = ""

        if USE_GUI:
            pygame.init()
            self.window_height = GRID_SIZE * CELL_SIZE + INPUT_BOX_HEIGHT + OUTPUT_BOX_HEIGHT
            self.window_size = GRID_SIZE * CELL_SIZE
            self.screen = pygame.display.set_mode((self.window_size, self.window_height))
            pygame.display.set_caption("L-Game Grid")

            # Input and output box settings
            self.input_box = pygame.Rect(0, GRID_SIZE * CELL_SIZE, self.window_size, INPUT_BOX_HEIGHT)
            self.output_box = pygame.Rect(0, GRID_SIZE * CELL_SIZE + INPUT_BOX_HEIGHT, self.window_size, OUTPUT_BOX_HEIGHT)
            self.font = pygame.font.Font(None, 32)

    def draw_grid(self):
        """Draw the game grid onto the Pygame screen based on the current grid state."""
        if not USE_GUI:
            print(self.state.render())
            return

        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                cell_value = self.state.grid.grid[y, x]
                color = COLORS.get(cell_value, (0, 0, 0))
                pygame.draw.rect(
                    self.screen,
                    color,
                    pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                )
                pygame.draw.rect(
                    self.screen,
                    (0, 0, 0),
                    pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                    1
                )

    def draw_input_box(self):
        """Draw the input box for user commands."""
        pygame.draw.rect(self.screen, (200, 200, 200), self.input_box)  
        text_surface = self.font.render(self.input_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, GRID_SIZE * CELL_SIZE + 10))

    def draw_output_box(self):
        """Draw the output box to display messages."""
        pygame.draw.rect(self.screen, (240, 240, 240), self.output_box) 

        font = pygame.font.Font(None, 24)
        line_height = font.get_linesize()

        start_index = max(0, len(self.messages) - VISIBLE_MESSAGES)
        displayed_messages = self.messages[start_index:]

        for i, message in enumerate(displayed_messages):
            message_surface = font.render(message, True, (0, 0, 0))
            self.screen.blit(message_surface, (10, GRID_SIZE * CELL_SIZE + INPUT_BOX_HEIGHT + 5 + i * line_height))

    def add_message(self, message: str):
        """Add a message to the output box."""
        self.messages.append(message)

    def show_legal_moves(self):
        """Display legal moves in the output box for the current agent's turn."""
        self.add_message("Legal moves:")
        moves = (
            self.state.grid.get_red_legal_moves()
            if self.turn == 0
            else self.state.grid.get_blue_legal_moves()
        )
        for move in moves:
            self.add_message(f"  {move}")

    def process_command(self, command: str):
        """Process commands entered by the user."""
        command = command.lower().strip()

        if command == "help":
            self.add_message("Help: Enter moves in the format 'x y direction'.")
            self.add_message("For example, '1 2 N' for (x=1, y=2) with North orientation.")
            self.add_message("Type 'legal' to view legal moves or 'exit' to quit.")
        elif command == "legal":
            self.show_legal_moves()
        elif command == "exit":
            pygame.quit()
            exit()
        else:
            try:
                action = self.parse_move_command(command)
                if action in self.state.get_legal_actions(self.turn):
                    self.state = self.state.generate_successor(action, self.turn)
                    self.add_message(f"Player {self.turn + 1} moved: {command}")
                    self.turn = 1 - self.turn
                else:
                    self.add_message("Invalid move. Please enter a legal move.")
            except ValueError as e:
                self.add_message(f"Error: {e}")

    def parse_move_command(self, command: str):
        """Parse the command string and return an LGameAction."""
        args = command.split()
        if len(args) == 3:
            x, y, direction = int(args[0]), int(args[1]), args[2].upper()
            return LGameAction(LPiecePosition(Coordinate(x - 1, y - 1), Orientation(direction)))
        else:
            raise ValueError("Invalid move format. Type 'help' for instructions.")

    def update_display(self):
        """Clear the screen and redraw the grid, input box, and output box."""
        if not USE_GUI:
            return

        self.screen.fill((255, 255, 255)) 
        self.draw_grid()
        self.draw_input_box()
        self.draw_output_box()
        pygame.display.flip()

    def run(self):
        """Main game loop with support for both AI and human turns."""
        running = True

        while running:
            self.update_display()

            # Check if game is over
            if self.state.is_terminal():
                winner = 1 - self.turn
                self.add_message(f"Player {winner + 1} wins!")
                self.update_display()
                pygame.display.flip()
                pygame.time.delay(4000)
                break

            current_agent = self.state.agents[self.turn]
            legal_moves = self.state.get_legal_actions(self.turn)

            if not legal_moves:
                winner = 1 - self.turn
                self.add_message(f"Player {winner + 1} wins!")
                self.update_display()
                pygame.display.flip()
                pygame.time.delay(4000)
                break

            if isinstance(current_agent, HumanAgent):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RETURN:
                            self.process_command(self.input_text)
                            self.input_text = ""
                        elif event.key == pygame.K_BACKSPACE:
                            self.input_text = self.input_text[:-1]
                        else:
                            self.input_text += event.unicode
            else:

                action = current_agent.get_action(self.state)
                self.state = self.state.generate_successor(action, self.turn)
                self.add_message(f"Player {self.turn + 1} (AI) moved: {action}")
                self.turn = 1 - self.turn
                pygame.time.delay(100)  # Delay for AI moves edit or disable this glass for higher depths

        pygame.quit()
        print("Game has ended.")
