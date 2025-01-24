import pygame
from pygame.locals import QUIT, KEYDOWN, K_SPACE

from src.flappy_bird import FlappyBird


def play_with_keyboard():
    """
    Function to play Flappy Bird using keyboard (space key).
    """
    # Initialize pygame
    pygame.init()

    # Create the FlappyBird game instance
    game_state = FlappyBird()

    # Main game loop
    while True:
        # Get the current frame of the game
        action = 0  # Default action (do nothing)
        for event in pygame.event.get():
            if event.type == QUIT:  # Quit the game
                pygame.quit()
                return
            elif event.type == KEYDOWN and event.key == K_SPACE:  # Space key pressed
                action = 1  # Bird flaps (action = 1)

        # Advance the game by one frame with the given action
        _, reward, terminal = game_state.next_frame(action)
        print(f"Reward: {reward}, Terminal: {terminal}")

        # If the game ends (bird collides), reinitialize the game
        if terminal:
            print("Game over! Restarting...")
            game_state = FlappyBird()


if __name__ == "__main__":
    play_with_keyboard()
