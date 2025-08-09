import numpy as np
import tyro
import torch
import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

from src.flappy_bird import FlappyBird
from src.utils import get_device, save_np_as_image
from src.obs_processor import ObsProcessor
from src.dataset import HDF5DataSaver


@dataclass
class Args:
    """Implementation of Deep Q Network to play Flappy Bird - Testing"""

    image_size: int = 84
    """The common width and height for all images"""

    model_path: str = "outputs/trained_models/flappy_bird_800000"
    """Path to the trained model"""

    max_steps: int = 10000000000
    """Maximum steps per test episode"""

    save_data: bool = True
    """Save test data and results"""

    output_dir: str = "outputs/dataset"
    """Directory to save test results"""


class TestRunner:
    def __init__(
        self,
        model_path: str,
        image_size: int = 84,
        save_data: bool = False,
        output_dir: str = "test_results",
    ):
        self.model_path = model_path
        self.image_size = image_size
        self.save_data = save_data
        self.output_dir = output_dir
        self.data_saver = None

        if self.save_data:
            os.makedirs(self.output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = os.path.basename(self.model_path).replace(".", "_")
            filename = f"observations_actions_{model_name}_{timestamp}.h5"
            data_filepath = os.path.join(self.output_dir, filename)

            print(f"Data will be saved to: {data_filepath}")

            # Initialize HDF5 data saver
            self.data_saver = HDF5DataSaver(data_filepath, model_path)

        # Initialize model and game
        self.device = get_device()
        self.model = self._load_model()
        self.env = FlappyBird()

        # Initialize state processor
        self.state_processor = ObsProcessor(
            original_image_size=(self.env.screen_width, int(self.env.base_y)),
            target_image_size=self.image_size,
            device=self.device,
        )

        # Data collection
        self.test_data = {
            "model_path": model_path,
            "start_time": datetime.now().isoformat(),
            "steps_data": [],  # Store obs and actions for each step
        }

    def _load_model(self):
        """Load the trained model"""
        print(f"Loading model: {self.model_path}")
        model = torch.load(
            self.model_path,
            weights_only=False,
        )
        model.eval()
        model.to(self.device)
        return model

    def run_test(self, max_steps: Optional[int] = None, verbose: bool = True):
        """Run a single test episode"""
        image, reward, terminal = self.env.next_frame(0)
        state = self.state_processor.initialize_state(image)

        step_count = 0
        total_reward = 0

        while True:
            if max_steps and step_count >= max_steps:
                break

            # Get model prediction
            with torch.no_grad():
                prediction = self.model(state)[0]
                action = torch.argmax(prediction).item()

            # Take action
            next_image, reward, terminal = self.env.next_frame(action)
            next_state = self.state_processor.update_state(next_image)

            total_reward += reward
            step_count += 1

            if verbose and step_count % 100 == 0:
                print(
                    f"Step: {step_count}, Action: {action}, Reward: {reward}, Total Reward: {total_reward}"
                )

            # Save observation and action immediately
            if self.data_saver:
                self.data_saver.save_step_data(
                    step=step_count,
                    observation=image,
                    action=np.array(action),
                )

            if terminal:
                if verbose:
                    print(
                        f"Game Over! Steps: {step_count}, Total Reward: {total_reward}"
                    )
                # Finalize data file
                if self.data_saver:
                    final_score = getattr(self.env, "score", 0)
                    self.data_saver.finalize(step_count, total_reward, final_score)
                    print(f"Data saved to: {self.data_saver.filepath}")
                break

            state = next_state
            image = next_image

        return {
            "steps": step_count,
            "total_reward": total_reward,
            "final_score": getattr(self.env, "score", 0),
        }


def main():
    args = tyro.cli(Args)

    tester = TestRunner(
        model_path=args.model_path,
        image_size=args.image_size,
        save_data=args.save_data,
        output_dir=args.output_dir,
    )

    try:
        tester.run_test(args.max_steps)
    except KeyboardInterrupt:
        del tester


if __name__ == "__main__":
    main()
