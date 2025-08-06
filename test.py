"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import argparse
import torch
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import pickle
import numpy as np
import h5py

from src.deep_q_network import DeepQNetwork
from src.flappy_bird import FlappyBird
from src.utils import pre_processing
from data_saver import HDF5DataSaver


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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.game_state = FlappyBird()

        # Data collection
        self.test_data = {
            "model_path": model_path,
            "start_time": datetime.now().isoformat(),
            "steps_data": [],  # Store obs and actions for each step
        }

    def _load_model(self):
        """Load the trained model"""
        print(f"Loading model: {self.model_path}")
        if torch.cuda.is_available():
            model = torch.load(self.model_path)
        else:
            model = torch.load(
                self.model_path,
                map_location=lambda storage, loc: storage,
                weights_only=False,
            )
        model.eval()
        model.to(self.device)
        return model

    def _preprocess_image(self, image):
        """Preprocess game image"""
        processed = pre_processing(
            image[: self.game_state.screen_width, : int(self.game_state.base_y)],
            self.image_size,
            self.image_size,
        )
        return torch.from_numpy(processed).to(self.device)

    def _get_initial_state(self):
        """Get initial game state"""
        image, reward, terminal = self.game_state.next_frame(0)
        image = self._preprocess_image(image)
        state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]
        return state, reward, terminal

    def run_test(self, max_steps: Optional[int] = None, verbose: bool = True):
        """Run a single test episode"""
        state, reward, terminal = self._get_initial_state()

        step_count = 0
        total_reward = 0

        try:
            while True:
                if max_steps and step_count >= max_steps:
                    break

                # Get model prediction
                with torch.no_grad():
                    prediction = self.model(state)[0]
                    action = torch.argmax(prediction).item()

                # Save observation and action immediately
                if self.save_data and self.data_saver:
                    self.data_saver.save_step_data(
                        step=step_count,
                        observation=state.cpu().numpy(),
                        action=action,
                    )

                # Take action
                next_image, reward, terminal = self.game_state.next_frame(action)
                next_image = self._preprocess_image(next_image)
                next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]

                total_reward += reward
                step_count += 1

                if verbose and step_count % 100 == 0:
                    print(
                        f"Step: {step_count}, Action: {action}, Reward: {reward}, Total Reward: {total_reward}"
                    )

                if terminal:
                    if verbose:
                        print(
                            f"Game Over! Steps: {step_count}, Total Reward: {total_reward}"
                        )
                    # Finalize data file
                    if self.save_data and self.data_saver:
                        final_score = getattr(self.game_state, "score", 0)
                        self.data_saver.finalize(step_count, total_reward, final_score)
                        print(f"Data saved to: {self.data_saver.filepath}")
                    break

                state = next_state

        except Exception as e:
            # Ensure data is saved even if an error occurs
            if self.save_data and self.data_saver:
                final_score = getattr(self.game_state, "score", 0)
                self.data_saver.finalize(step_count, total_reward, final_score)
            raise e

        return {
            "steps": step_count,
            "total_reward": total_reward,
            "final_score": getattr(self.game_state, "score", 0),
        }


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Flappy Bird - Testing"""
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=84,
        help="The common width and height for all images",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="trained_models/flappy_bird_800000",
        help="Path to the trained model",
    )
    parser.add_argument(
        "--max_steps", type=int, default=None, help="Maximum steps per test episode"
    )
    parser.add_argument(
        "--save_data", action="store_true", help="Save test data and results"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_results",
        help="Directory to save test results",
    )

    return parser.parse_args()


def test_single_model(model_path: str, args):
    """Test a single model"""
    tester = TestRunner(
        model_path=model_path,
        image_size=args.image_size,
        save_data=args.save_data,
        output_dir=args.output_dir,
    )

    tester.run_test(args.max_steps)


def main():
    args = get_args()
    args.save_data = True

    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    test_single_model(args.model_path, args)


if __name__ == "__main__":
    main()
