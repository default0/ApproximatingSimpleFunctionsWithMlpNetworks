from typing import Callable
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
signal_size = 1

class Test:
    def __init__(self, function_to_approximate: Callable[[torch.Tensor], torch.Tensor], sample_inputs: Callable[[int], torch.Tensor], get_eval_samples: Callable[[], torch.Tensor]):
        self.function_to_approximate = function_to_approximate
        self.sample_inputs = sample_inputs
        self.get_eval_samples = get_eval_samples

    def get_inputs(self, num_inputs: int) -> torch.Tensor:
        return self.sample_inputs(num_inputs)
    
    def get_targets(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.function_to_approximate(inputs)
    
    def get_eval_targets(self) -> torch.Tensor:
        return self.function_to_approximate(self.get_eval_samples())

def square(x: torch.Tensor):
    return x ** 2

def square_root(x: torch.Tensor):
    return torch.sqrt(x)

def abs(x: torch.Tensor):
    return torch.abs(x)

def sin(x: torch.Tensor):
    return torch.sin(x)

def cos(x: torch.Tensor):
    return torch.cos(x)

def log(x: torch.Tensor):
    return torch.log(x)

def exp(x: torch.Tensor):
    return torch.exp(x)

def sample_zero_to_one(num_samples: int) -> torch.Tensor:
    samples = torch.abs(torch.rand(num_samples, signal_size, device=device))
    max_sample = torch.max(samples)
    return samples / max_sample

def sample_point01_to_one(num_samples: int) -> torch.Tensor:
    samples = torch.abs(torch.rand(num_samples, signal_size, device=device))
    samples += 0.01
    max_sample = torch.max(samples)
    return samples / max_sample

def sample_negative_one_to_one(num_samples: int) -> torch.Tensor:
    samples = torch.randn(num_samples, signal_size, device=device)
    max_sample = torch.max(samples)
    return samples / max_sample

def sample_negative_2pi_to_2pi(num_samples: int) -> torch.Tensor:
    return sample_negative_one_to_one(num_samples) * 2 * np.pi

# Define the MLP model
class SimpleMLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, processing_ratio: int):
        super(SimpleMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, input_size * processing_ratio, device=device), # Single linear layer
            nn.BatchNorm1d(input_size * processing_ratio, device=device),
            nn.ReLU(),
            nn.Linear(input_size * processing_ratio, output_size, device=device)
        )

    def forward(self, x):
        return self.fc(x)

def run_test(processing_ratio: int, test: Test):
    test_name = f"{test.function_to_approximate.__name__}_r{processing_ratio:03}"
    writer = SummaryWriter(f"runs/{test_name}")

    # Prepare the data
    num_batches = 128
    batch_size = 2048

    # Initialize the model, loss function, and optimizer
    model = SimpleMLP(signal_size, signal_size, processing_ratio)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 1000
    for epoch in range(num_epochs):
        inputs = test.get_inputs(num_batches * batch_size)
        targets = test.get_targets(inputs)

        for i in range(num_batches):
            input_batch = inputs[i * batch_size : (i + 1) * batch_size]
            target_batch = targets[i * batch_size : (i + 1) * batch_size]

            # Forward pass
            output = model(input_batch)
            loss = criterion(output, target_batch)
            if torch.isnan(loss):
                print("Loss is NaN")
                return

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        writer.add_scalar("Loss", loss.item(), epoch)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)
        # Decay learning rate
        if optimizer.param_groups[0]['lr'] > 1e-5:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.95
    
    writer.close()

    model.eval()

    # Evaluate the model
    # We do several evaluations:
    # 1) Cumulative L1 loss over 5 training batches
    # 2) Cumulative L1 loss over evaluation samples
    # 3) Cumulative MAPE loss over 5 training batches
    # 4) Cumulative MAPE loss over evaluation samples
    # 5) Plotting model outputs vs. targets over the evaluation samples
    # 6) Plotting L1 loss over the evaluation samples
    # 7) Plotting MAPE loss over the evaluation samples

    # We combine 5, 6 and 7 in the same plot for simplicity
    # 1-4 will simply be logged as scalar outputs
    # The plot for 5-7 will be saved as a figure (png)

    # Ensure the dir evals/test_name exists
    eval_dir = f"evals/{test_name}"
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    results = dict()

    # 1) Cumulative L1 loss over 5 training batches
    inputs = test.get_inputs(5 * batch_size)
    targets = test.get_targets(inputs)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    results["train_l1_loss"] = loss.item() * batch_size

    # 2) Cumulative L1 loss over evaluation samples
    eval_inputs = test.get_eval_samples().view(-1, signal_size)
    eval_targets = test.get_eval_targets().view(-1, signal_size)
    eval_outputs = model(eval_inputs)
    loss = criterion(eval_outputs, eval_targets)
    results["eval_l1_loss"] = loss.item() * eval_inputs.shape[0]

    # 3) Cumulative MAPE loss over 5 training batches
    inputs = test.get_inputs(5 * batch_size)
    targets = test.get_targets(inputs)
    outputs = model(inputs)
    mape = torch.abs((outputs - targets) / torch.clamp(torch.abs(targets), min=1e-06)).sum()
    results["train_mape_loss"] = mape.item()

    # 4) Cumulative MAPE loss over evaluation samples
    eval_inputs = test.get_eval_samples().view(-1, signal_size)
    eval_targets = test.get_eval_targets().view(-1, signal_size)
    eval_outputs = model(eval_inputs)
    mape = torch.abs((eval_outputs - eval_targets) / torch.clamp(torch.abs(eval_targets), min=1e-06)).sum()
    results["eval_mape_loss"] = mape.item()

    # 5) Plotting model outputs vs. targets over the evaluation samples
    eval_inputs = test.get_eval_samples().view(-1, signal_size)
    eval_targets = test.get_eval_targets().view(-1, signal_size)
    eval_outputs = model(eval_inputs)
    l1 = torch.abs(eval_outputs - eval_targets)

    plt.figure()

    plt.plot(eval_inputs.cpu().numpy(), eval_targets.cpu().numpy(), label="Target")
    plt.plot(eval_inputs.cpu().numpy(), eval_outputs.cpu().detach().numpy(), label="Output")
    plt.plot(eval_inputs.cpu().numpy(), l1.cpu().detach().numpy(), label="L1 Loss")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.title(f"{test_name}")
    plt.legend()
    plt.savefig(f"{eval_dir}/outputs_vs_targets.png")

    # Dump results to a json file (pretty-printed)
    with open(f"{eval_dir}/results.json", "w") as f:
        json.dump(results, f, indent=4)

tests = [
    Test(square, sample_negative_one_to_one, lambda: torch.linspace(-2, 2, 1000, device=device)),
    Test(square_root, sample_zero_to_one, lambda: torch.linspace(0, 2, 1000, device=device)),
    Test(abs, sample_negative_one_to_one, lambda: torch.linspace(-2, 2, 1000, device=device)),
    Test(sin, sample_negative_2pi_to_2pi, lambda: torch.linspace(-4 * np.pi, 4 * np.pi, 1000, device=device)),
    Test(cos, sample_negative_2pi_to_2pi, lambda: torch.linspace(-4 * np.pi, 4 * np.pi, 1000, device=device)),
    Test(log, sample_point01_to_one, lambda: torch.linspace(0.001, 2, 1000, device=device)),
    Test(exp, sample_negative_one_to_one, lambda: torch.linspace(-2, 2, 1000, device=device))
]

processing_ratios = [1, 2, 4, 8, 16, 32, 64, 128]
for test in tests:
    for processing_ratio in processing_ratios:
        run_test(processing_ratio, test)