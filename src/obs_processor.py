import torch
from src.utils import pre_processing


class ObsProcessor:
    """
    Handles preprocessing and state management between environment and model.
    Maintains a rolling history of processed frames for the DQN input.
    """

    def __init__(
        self,
        target_image_size: int = 84,
        stack_size: int = 4,
        original_image_size: tuple = (288, 512),
        device: torch.device = None,
    ):
        self.target_image_size = target_image_size
        self.stack_size = stack_size
        self.original_image_size = original_image_size
        self.device = device
        self.current_state = None

    def _preprocess_image(self, image):
        """Preprocess a single frame from the environment"""
        # Assert image dimensions match expected values
        assert (
            image.shape[0] >= self.original_image_size[0]
        ), f"Image width {image.shape[0]} is smaller than expected {self.original_image_size[0]}"
        assert image.shape[1] >= int(
            self.original_image_size[1]
        ), f"Image height {image.shape[1]} is smaller than expected {int(self.original_image_size[1])}"

        processed = pre_processing(
            image[: self.original_image_size[0], : int(self.original_image_size[1])],
            self.target_image_size,
            self.target_image_size,
        )
        return torch.from_numpy(processed).to(self.device)

    def initialize_state(self, initial_image):
        """Initialize the state stack with the first frame repeated"""
        processed_image = self._preprocess_image(initial_image)

        # Stack the same frame 4 times for initial state
        self.current_state = torch.cat(
            tuple(processed_image for _ in range(self.stack_size))
        )[None, :, :, :]
        return self.current_state

    def update_state(self, new_image):
        """Update state by adding new frame and removing oldest frame"""
        if self.current_state is None:
            raise RuntimeError("State not initialized. Call initialize_state() first.")

        new_processed_image = self._preprocess_image(new_image)
        # Remove oldest frame and add new frame
        self.current_state = torch.cat(
            (self.current_state[0, 1:, :, :], new_processed_image)
        )[None, :, :, :]
        return self.current_state

    def get_current_state(self):
        """Get the current state"""
        return self.current_state

    def reset(self):
        """Reset the processor state"""
        self.current_state = None
