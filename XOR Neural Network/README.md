# XOR Neural Network Experiments in PyTorch

This project implements and analyzes multiple experiments training a neural network to learn the XOR function using PyTorch.

## Experiments Overview

A series of 9 experiments were conducted to study the effects of:

- **Learning Rates:** 0.1 and 0.01
- **Hidden Units:** 1, 2, and 4 neurons
- **Bypass Connection:** Enabled and disabled
- **Additional Custom Experiment:** 1 hidden neuron with bypass and a learning rate of your choice

## Training Details

- Loss function: Cross-Entropy Loss
- Optimizer: Gradient Descent
- Training and validation performed on predefined datasets
- Each experiment repeated 10 times with different random initializations

## Stopping Criteria

Training was stopped if:

1. Validation loss < 0.2 and improvement < 0.0001 over the last 10 epochs (successful run)
2. 40,000 epochs reached without satisfying criterion (failed run)

## Metrics Recorded

For each experiment:

- Average and standard deviation of:
  - Number of epochs to stopping
  - Final train loss
  - Final validation loss
- Count of failed runs

Additionally:

- Truth tables showing neuron output for all input combinations
- Analysis of whether the trained neuron approximates any logical function

## Analysis Conducted

- Impact of the number of hidden units on convergence speed
- Effect of the bypass connection on training
- Influence of learning rate on variance of epochs required
- Visualization of hidden neuron outputs and behaviors
