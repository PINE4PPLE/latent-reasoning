from typing import Dict
import numpy as np
import torch
import torch.nn as nn

import ray

ds = ray.data.from_numpy(np.ones((256, 100)))

class TorchPredictor:
    def __init__(self):
        # Move the neural network to GPU device by specifying "cuda".
        self.model = nn.Sequential(
            nn.Linear(in_features=100, out_features=1),
            nn.Sigmoid(),
        ).cuda()
        self.model.eval()

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # Move the input batch to GPU device by specifying "cuda".
        tensor = torch.as_tensor(batch["data"], dtype=torch.float32, device="cuda")
        with torch.inference_mode():
            # Move the prediction output back to CPU before returning.
            return {"output": self.model(tensor).cpu().numpy()}

# Use 2 actors, each actor using 1 GPU. 2 GPUs total.
predictions = ds.map_batches(
    TorchPredictor,
    num_gpus=1,
    # Specify the batch size for inference.
    # Increase this for larger datasets.
    batch_size=1,
    # Set the concurrency to the number of GPUs in your cluster.
    concurrency=8,
    )
predictions.show(limit=1)