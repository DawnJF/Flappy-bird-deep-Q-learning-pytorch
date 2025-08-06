import h5py
import numpy as np
import os
from datetime import datetime


class HDF5DataSaver:
    def __init__(self, filepath: str, model_path: str):
        self.filepath = filepath
        self.model_path = model_path
        self.file = None
        self.step_count = 0
        self.datasets = {}

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Open HDF5 file and initialize datasets
        self._initialize_file()

    def _initialize_file(self):
        """Initialize HDF5 file and datasets"""
        self.file = h5py.File(self.filepath, "w")

        # Save metadata
        metadata_group = self.file.create_group("metadata")
        metadata_group.attrs["model_path"] = self.model_path
        metadata_group.attrs["start_time"] = datetime.now().isoformat()

        # Create datasets with unlimited size along first dimension
        # We'll determine the observation shape from the first data point
        self.observations_initialized = False

        # Create action dataset (1D integers)
        self.datasets["actions"] = self.file.create_dataset(
            "actions", (0,), maxshape=(None,), dtype=np.int32, chunks=True
        )

        # Create step numbers dataset
        self.datasets["steps"] = self.file.create_dataset(
            "steps", (0,), maxshape=(None,), dtype=np.int32, chunks=True
        )

    def save_step_data(self, step: int, observation: np.ndarray, action: np.ndarray):
        """Save data for one step"""
        if not self.observations_initialized:
            # Initialize observations dataset with proper shape
            obs_shape = observation.shape
            self.datasets["observations"] = self.file.create_dataset(
                "observations",
                (0,) + obs_shape[1:],  # Skip batch dimension
                maxshape=(None,) + obs_shape[1:],
                dtype=np.float32,
                chunks=True,
            )
            self.observations_initialized = True

        # Resize datasets to accommodate new data
        self.datasets["steps"].resize((self.step_count + 1,))
        self.datasets["actions"].resize((self.step_count + 1,))
        self.datasets["observations"].resize(
            (self.step_count + 1,) + self.datasets["observations"].shape[1:]
        )

        # Save data
        self.datasets["steps"][self.step_count] = step
        self.datasets["actions"][self.step_count] = action
        self.datasets["observations"][self.step_count] = observation[
            0
        ]  # Remove batch dimension

        self.step_count += 1

        # Flush to disk periodically for safety
        if self.step_count % 100 == 0:
            self.file.flush()

    def finalize(self, total_steps: int, total_reward: float, final_score: int):
        """Finalize the file with summary information"""
        if self.file:
            # Add final metadata
            self.file.attrs["end_time"] = datetime.now().isoformat()
            self.file.attrs["total_steps"] = total_steps
            self.file.attrs["total_reward"] = total_reward
            self.file.attrs["final_score"] = final_score

            # Final flush and close
            self.file.flush()
            self.file.close()
            self.file = None

    def __del__(self):
        """Ensure file is closed"""
        if self.file:
            self.file.close()
