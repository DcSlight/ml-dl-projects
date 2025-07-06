import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.weight = nn.Parameter(torch.rand([self.out_features, self.in_features]))
        if self.bias is not None:
            self.bias = nn.Parameter(torch.rand([self.out_features]))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.matmul(input, torch.transpose(self.weight, 0, 1)) + self.bias

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class BTU(torch.nn.Module):
    def __init__(self, T=0.2, inplace: bool = False):
        super(BTU, self).__init__()
        self.T = T

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 1 / (1 + torch.exp(-input / self.T))


def Loss(out, t_train):
    return -torch.sum(t_train * torch.log(out) + (1.0 - t_train) * torch.log(1.0 - out)) / out.size()[
        0]  # Cross Entropy loss function


class XOR_Net_Model(nn.Module):
    def __init__(self, dim, num_hidden, out_dim, T=0.5, bypass=True):
        super().__init__()
        self.bypass = bypass
        self.hidden = Linear(dim, num_hidden)
        if self.bypass:
            self.output = Linear(num_hidden + dim, out_dim)
        else:
            self.output = Linear(num_hidden, out_dim)
        self.BTU = BTU(T)

    def forward(self, input):
        z1 = self.hidden(input)
        y1 = self.BTU(z1)
        if self.bypass:
            y1_concat = torch.cat((input, y1), 1)
            z2 = self.output(y1_concat)
        else:
            z2 = self.output(y1)
        return self.BTU(z2)


def train(model, x_train, t_train, optimizer):
    y_pred = model(x_train)
    loss = Loss(y_pred, t_train)

    # zero gradients berfore running the backward pass
    optimizer.zero_grad()

    # backward pass to compute the gradient of loss
    # backprop + accumulate
    loss.backward()

    # update params
    optimizer.step()
    return loss


# define test step operation:
def test(model, x_test, t_test):
    loss = Loss(model(x_test), t_test)
    return loss


# Function to calculate statistics and print experiment results
def pretty_print_experiment(results, failed_experiments):
    # Convert results to a DataFrame for detailed experiment stats
    df = pd.DataFrame(results, columns=[
        "Experiment", "Mean Train Loss", "Std Dev Train Loss",
        "Epochs Until Stop", "Final Train Loss", "Final Val Loss"
    ])
    print("\nExperiment Results Table:\n")
    print(df.to_string(index=False))

    # Summary Table for percentage std deviation and mean
    summary_data = {
        "Metric": [
            "Epochs Until Stop",
            "Final Train Loss",
            "Final Val Loss"
        ],
        "Mean": [
            df["Epochs Until Stop"].mean(),
            df["Final Train Loss"].mean(),
            df["Final Val Loss"].mean()
        ],
        "Std Dev %": [
            (df["Epochs Until Stop"].std() / df["Epochs Until Stop"].mean()) * 100,
            (df["Final Train Loss"].std() / df["Final Train Loss"].mean()) * 100,
            (df["Final Val Loss"].std() / df["Final Val Loss"].mean()) * 100
        ]
    }
    summary_df = pd.DataFrame(summary_data)

    print("\nSummary Table (Percentage Std Dev and Mean):\n")
    print(summary_df.to_string(index=False))
    print(f"\nTotal Failures: {failed_experiments}")


def print_hidden_truth_table(model, x_train):
    with torch.no_grad():  # Disable gradient calculation for evaluation
        raw_hidden_outputs = model.hidden(x_train)  # Outputs before activation
        activated_hidden_outputs = model.BTU(raw_hidden_outputs)  # Outputs after activation
        print("\nTruth Table for Hidden Neuron:")
        print(f"{'Input':<20} {'Hidden Output':<20}")
        print("-" * 60)
        for i, inp in enumerate(x_train):
            print(f"{str(inp.tolist()):<20} {str(activated_hidden_outputs[i].tolist()):<20}")


# Function to perform a single experiment loop
def run_experiments(
        x_train, t_train, x_val, t_val,
        num_hidden, bypass, l_rate,
        num_epochs=40000, threshold=0.0001, success_limit=10,
        truth_table_function=None
):
    num_successes = 0  # Counter for successful experiments
    experiment_count = 0  # Track total experiments
    failed_experiments = 0  # Count of failed experiments
    results = []  # To store experiment data
    while num_successes < success_limit:
        experiment_count += 1
        print(f"\nStarting Experiment {experiment_count}...\n")

        # Initialize model and optimizer for each experiment
        model = XOR_Net_Model(dim=dim, num_hidden=num_hidden, out_dim=out_dim, bypass=bypass)
        optimizer = torch.optim.SGD(model.parameters(), lr=l_rate)

        best_val_loss = 1e6  # Large initial value
        no_improve_epochs = 0

        train_losses = []  # To track train losses for stats
        stopped_epoch = 0

        for epoch in range(num_epochs):
            # Train on the training set
            train_loss = train(model, x_train, t_train, optimizer)
            train_losses.append(train_loss.item())

            # Evaluate on the validation set
            val_loss = test(model, x_val, t_val)

            # Check for validation loss improvement
            if best_val_loss - val_loss > threshold:
                best_val_loss = val_loss
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            # Stop criteria (based on your condition)
            if val_loss < 0.2 and no_improve_epochs >= 10:
                print(f"Experiment {experiment_count}: Stopping early: Validation loss < 0.2 at epoch {epoch}")
                num_successes += 1  # Count this as a successful experiment
                stopped_epoch = epoch
                break

            # Print progress every 2000 epochs
            if epoch % 2000 == 0:
                print(f"Epoch {epoch}, Train Loss: {train_loss.item()}, Val Loss: {val_loss.item()}")

        # Optionally print the truth table for the hidden neuron
        if truth_table_function:
            truth_table_function(model, x_train)

        # Record experiment results
        if stopped_epoch == 0:  # If the loop never stops successfully
            failed_experiments += 1
            print(f"The Experiment failed - After {num_epochs} epochs ")
        else:
            results.append([
                experiment_count,
                np.mean(train_losses),
                np.std(train_losses),
                stopped_epoch,
                train_losses[-1],  # Final train loss
                val_loss.item()  # Final validation loss
            ])

    return results, failed_experiments


dim = 2
out_dim = 1

# Training set (original XOR examples)
x_train = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
t_train = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Validation set (new examples provided)
x_val = torch.tensor([[1, 0.1], [1, 0.9], [0.1, 1], [0.9, 0.9], [0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
t_val = torch.tensor([[1], [0], [1], [0], [0], [1], [1], [0]], dtype=torch.float32)

configs = [
    {"num_hidden": 2, "bypass": True, "l_rate": 0.1},
    {"num_hidden": 2, "bypass": False, "l_rate": 0.1},
    {"num_hidden": 4, "bypass": True, "l_rate": 0.1},
    {"num_hidden": 4, "bypass": False, "l_rate": 0.1},
    {"num_hidden": 2, "bypass": True, "l_rate": 0.01},
    {"num_hidden": 2, "bypass": False, "l_rate": 0.01},
    {"num_hidden": 4, "bypass": True, "l_rate": 0.01},
    {"num_hidden": 4, "bypass": False, "l_rate": 0.01},
    {"num_hidden": 1, "bypass": True, "l_rate": 0.1, "truth_table_function": print_hidden_truth_table}
]

results = []  # To store detailed results for analysis

for i, config in enumerate(configs, start=1):
    print(f"\n-------------------------------------------------------------------------")
    print(f"Running Configuration {i}: {config}\n")
    experiment_results, failed_experiments = run_experiments(
        x_train=x_train,
        t_train=t_train,
        x_val=x_val,
        t_val=t_val,
        num_hidden=config["num_hidden"],
        bypass=config["bypass"],
        l_rate=config["l_rate"],
        success_limit=10,
        truth_table_function=config.get("truth_table_function")
    )
    pretty_print_experiment(experiment_results, failed_experiments)

    # Collect data for analysis
    for result in experiment_results:
        results.append({
            "num_hidden": config["num_hidden"],
            "bypass": config["bypass"],
            "l_rate": config["l_rate"],
            "epochs": result[3],
        })

results_df = pd.DataFrame(results)

# Plot Graphs
plt.figure(figsize=(12, 6))

# 1. num_hidden vs average epochs
plt.subplot(1, 3, 1)
for bypass in [True, False]:
    subset = results_df[results_df["bypass"] == bypass]
    averages = subset.groupby("num_hidden")["epochs"].mean()
    plt.plot(averages.index, averages.values, label=f"Bypass={bypass}")
plt.title("Average Epochs vs Num Hidden Units")
plt.xlabel("Number of Hidden Units")
plt.ylabel("Average Epochs")
plt.legend()

# 2. bypass vs average epochs
plt.subplot(1, 3, 2)
bypass_averages = results_df.groupby("bypass")["epochs"].mean()
bypass_averages.plot(kind="bar", color=["blue", "orange"], rot=0)
plt.title("Average Epochs vs Bypass")
plt.xlabel("Bypass")
plt.ylabel("Average Epochs")

# 3. learning rate vs STD of epochs
plt.subplot(1, 3, 3)
stds = results_df.groupby("l_rate")["epochs"].std()
plt.plot(stds.index, stds.values, marker="o")
plt.title("STD of Epochs vs Learning Rate")
plt.xlabel("Learning Rate")
plt.ylabel("STD of Epochs")

plt.tight_layout()
plt.show()
