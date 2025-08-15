import numpy as np
from tqdm import tqdm
import tyro
import torch
import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

from src.net.jepa_thinking import JepaThinking
from src.net.deep_q_network import DeepQNetwork
from src.net.thinking import Thinking
from src.flappy_bird import FlappyBird
from src.utils import get_device, load_model, save_np_as_image
from src.obs_processor import ObsProcessor
from src.dataset import HDF5DataSaver


@dataclass
class Args:
    """Implementation of Deep Q Network to play Flappy Bird - Testing"""

    image_size: int = 84
    """The common width and height for all images"""

    # model_path: str = "outputs/trained_models/dqn_2000000"
    # model_path: str = "outputs/dqn/flappy_bird_1000000"
    # model_name = DeepQNetwork
    # model_path: str = "outputs/supervised/train_2025_0815_170228/best_model_2000.pth"
    # model_name = Thinking
    model_path: str = "outputs/compare/train_2025_0815_165647/final_model_4000.pth"
    model_name = JepaThinking

    max_steps: int = 10000000000
    """Maximum steps per test episode"""

    save_data: bool = False
    """Save test data and results"""

    output_dir: str = "outputs/dataset"
    """Directory to save test results"""


class TestRunner:
    def __init__(
        self,
        args,
    ):
        self.args = args
        self.model_path = args.model_path
        self.image_size = args.image_size
        self.save_data = args.save_data
        self.output_dir = args.output_dir
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
            stack_size=self.model.config.get("channel_dim", 4),
            original_image_size=(self.env.screen_width, int(self.env.base_y)),
            target_image_size=self.image_size,
            device=self.device,
        )

        # Data collection
        self.test_data = {
            "model_path": self.model_path,
            "start_time": datetime.now().isoformat(),
            "steps_data": [],  # Store obs and actions for each step
        }

    def _load_model(self):
        """Load the trained model"""
        print(f"Loading model: {self.model_path}")
        model = load_model(self.args.model_name, self.model_path)
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


def evaluate_models(model_info_list, num_episodes=10, max_steps=None, verbose=False):
    """
    Evaluate a list of models, each for num_episodes, and return average steps for each.

    Args:
        model_info_list: List of tuples (model_name, model_path)
        num_episodes: Number of episodes per model
        max_steps: Max steps per episode
        verbose: Print progress

    Returns:
        List of dicts: [{'model_name': ..., 'model_path': ..., 'avg_steps': ...}, ...]
    """
    results = []
    for model_name, model_path in tqdm(model_info_list):
        args = Args()
        args.model_name = model_name
        args.model_path = model_path
        args.save_data = False  # 不保存数据
        tester = TestRunner(args)
        steps_list = []
        for i in range(num_episodes):
            if verbose:
                print(f"Model: {model_name}, Episode: {i+1}/{num_episodes}")
            res = tester.run_test(max_steps=max_steps, verbose=False)
            steps_list.append(res["steps"])
        avg_steps = np.mean(steps_list)
        results.append(
            {
                "model_name": str(model_name),
                "model_path": str(model_path),
                "avg_steps": float(avg_steps),
            }
        )
        if verbose:
            print(f"Model: {model_name}/{model_path}, Avg Steps: {avg_steps}")
    return results


def main():
    args = tyro.cli(Args)

    tester = TestRunner(args)

    try:
        tester.run_test(args.max_steps)
    except KeyboardInterrupt:
        del tester


def test_models():
    model_info_list = [
        (JepaThinking, "outputs/compare/train_2025_0815_165647/checkpoint_1000.pth"),
        (JepaThinking, "outputs/compare/train_2025_0815_165647/checkpoint_2000.pth"),
        (JepaThinking, "outputs/compare/train_2025_0815_165647/checkpoint_3000.pth"),
        (JepaThinking, "outputs/compare/train_2025_0815_165647/checkpoint_4000.pth"),
        (Thinking, "outputs/compare/train_2025_0815_170228/checkpoint_1000.pth"),
        (Thinking, "outputs/compare/train_2025_0815_170228/checkpoint_2000.pth"),
        (Thinking, "outputs/compare/train_2025_0815_170228/checkpoint_3000.pth"),
        (Thinking, "outputs/compare/train_2025_0815_170228/checkpoint_4000.pth"),
        # ("Model3", "path/to/model3"),
    ]

    result = evaluate_models(model_info_list)
    print("Evaluation Results:")
    for res in result:
        print(
            f"Model: {res['model_name']}/{res['model_path']}, Avg Steps: {res['avg_steps']}"
        )


if __name__ == "__main__":
    # main()
    test_models()
