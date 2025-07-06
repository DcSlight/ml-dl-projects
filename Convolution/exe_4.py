import torch
from torch import nn
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import time
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, balanced_accuracy_score
import pandas as pd
from torchsummary import summary

activation = {}  ## global


def init():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    image_normalize = lambda x: x / 255.
    # one_hot = lambda t : nn.functional.one_hot(t) # not needed!
    train_loader_original = torch.utils.data.DataLoader(datasets.MNIST('/files/',
                                                                       train=True,
                                                                       download=True,
                                                                       transform=transforms.Compose(
                                                                           [transforms.ToTensor(),
                                                                            transforms.Lambda(
                                                                                image_normalize)])),
                                                        batch_size=100,
                                                        shuffle=True)
    train_images = train_loader_original.dataset.data / 255.  # save for later plotting
    train_labels = train_loader_original.dataset.targets
    # train_loader.dataset.targets = nn.functional.one_hot(train_loader.dataset.targets)

    test_loader = torch.utils.data.DataLoader(datasets.MNIST('/files/',
                                                             train=False,
                                                             download=True,
                                                             transform=transforms.Compose([transforms.ToTensor(),
                                                                                           transforms.Lambda(
                                                                                               image_normalize)])),
                                              batch_size=100,
                                              shuffle=True)

    # test_loader.dataset.targets = nn.functional.one_hot(test_loader.dataset.targets)

    print(train_labels.size())
    print(train_labels)
    print(train_images.size())
    print(train_images)
    return device, train_loader_original, train_images, train_labels, test_loader


class PrintLayer(nn.Module):
    def __init__(self, layer_out, image=0):
        super(PrintLayer, self).__init__()
        self.image = image
        self.layer_out = layer_out

    def forward(self, x):
        # Do your print / debug stuff here
        # imgplot = plt.imshow(x.cpu().detach().numpy()[0][0])

        self.layer_out = torch.clone(x[self.image])
        return x  # forward needs to be transparent


# Split
def split_data(train_loader_original, train_ratio=0.9, minibatch=50):
    # Total number of samples in the training dataset
    total_train_samples = len(train_loader_original.dataset)

    # Define the split sizes
    train_size = int(train_ratio * total_train_samples)
    val_size = total_train_samples - train_size

    # Split the training dataset into training and validation subsets
    train_subset, val_subset = random_split(train_loader_original.dataset, [train_size, val_size])

    # Create new DataLoaders for the training and validation subsets
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=minibatch, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=minibatch, shuffle=True)

    # Verify the sizes
    print(f"Training dataset size: {len(train_subset)}")
    print(f"Validation dataset size: {len(val_subset)}")
    return train_loader, val_loader


def accuracy(y, t):
    score, predicted = torch.max(y, 1)
    acc = (t == predicted).sum().float() / len(t)
    return acc


def model_training(model, train_loader, num_epochs=2, lr=0.001, max_batches=13000):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss()
    start_time = time.time()
    # Set the maximum number of batches
    batch_count = 0

    for epoch in range(num_epochs):
        print("epoch ", epoch + 1, ":")
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            # Increment batch count
            batch_count += 1

            # Stop if batch count exceeds max_batches
            if batch_count > max_batches:
                print("Reached 13,000 batches, stopping training.")
                break
            predictions = model(images)
            cross_entropy = loss(predictions, labels)

            if not (batch_idx % 100):
                print("CE: %.6f, ACC: %.3f" % (cross_entropy.item(), accuracy(predictions, labels).item()))

            # Zero gradients before running the backward pass
            optimizer.zero_grad()

            # Backward pass to compute the gradient of loss
            cross_entropy.backward()

            # Update parameters
            optimizer.step()

        # Stop the outer loop if max_batches has been reached
        if batch_count > max_batches:
            break

    # Final output of the training process
    print("Final train step - CE: %.6f, ACC: %.3f" % (cross_entropy.item(), accuracy(predictions, labels).item()))
    # Record the end time after training
    end_time = time.time()

    # Calculate the total time taken
    training_time = end_time - start_time

    # Print the total training time
    print(f"Training completed in {training_time:.2f} seconds")


def predict(model, dataset):
    predictions = []
    labels = []
    for batch_idx, (images, label) in enumerate(dataset):
        images, label = images.to(device), label.to(device)
        prediction = torch.argmax(model(images), 1)
        predictions.extend(prediction.cpu().numpy())
        labels.extend(label.cpu().numpy())
    return predictions, labels


def score(predictions, labels):
    # Convert predictions and labels to numpy arrays (if they aren't already)
    predictions = np.array(predictions)
    labels = np.array(labels)

    # Calculate recall, precision, F1 score, and accuracy for each target
    recall = recall_score(labels, predictions, average=None)  # Recall for each class
    precision = precision_score(labels, predictions, average=None)  # Precision for each class
    f1 = f1_score(labels, predictions, average=None)  # F1 score for each class
    accuracy_per_class = []
    for i in range(len(np.unique(labels))):  # Iterate over each class
        class_mask = (labels == i)  # Create a mask for the current class
        correct_class_predictions = np.sum(predictions[class_mask] == labels[class_mask])
        total_class_samples = np.sum(class_mask)
        accuracy_per_class.append(correct_class_predictions / total_class_samples)
    balanced_accuracy = balanced_accuracy_score(labels, predictions)  # Balanced accuracy

    # Print scores for each class
    print(f"Recall for each class: {recall}")
    print(f"Precision for each class: {precision}")
    print(f"F1 score for each class: {f1}")
    print(f"Accuracy for the dataset: {accuracy_per_class}")
    print(f"Balanced Accuracy for the dataset: {balanced_accuracy}")

    # return recall, precision, f1, accuracy, balanced_accuracy


def print_scores(model, train_loader, val_loader, test_loader, title):
    print("------------------------------------------")
    print(title)
    print("-------------------Train------------------")
    train_pred, train_labels = predict(model, train_loader)
    score(train_pred, train_labels)
    print("-------------------Validation------------------")
    val_pred, val_labels = predict(model, val_loader)
    score(val_pred, val_labels)
    print("-------------------Test------------------")
    test_pred, test_labels = predict(model, test_loader)
    score(test_pred, test_labels)
    print("------------------------------------------")


def init_conv2d_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        m.weight = nn.Parameter(torch.abs(m.weight))
        m.bias.data.fill_(0.01)


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


def first_model(train_loader_original):
    print("--------------------------- Model 1 --------------------------")
    # First Architecture - Split:
    train_loader, val_loader = split_data(train_loader_original)

    model1 = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 10)
    )

    ##First Architecture - Model training:
    model_training(model1, train_loader, num_epochs=2, lr=0.001, max_batches=13000)

    title = "First Architecture - No Hidden Layers:"
    print_scores(model1, train_loader, val_loader, test_loader, title)


def second_model(train_loader_original):
    print("--------------------------- Model 2 --------------------------")
    model2 = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 200),
        nn.ReLU(),
        nn.Linear(200, 200),
        nn.ReLU(),
        nn.Linear(200, 10)
    )

    # Second Architecture - Split:
    train_loader, val_loader = split_data(train_loader_original)

    # Second Architecture - model training:
    model_training(model2, train_loader, num_epochs=2, lr=0.001, max_batches=13000)

    title = "Second Architecture - 2 Hidden Layers FC 200 Neurons:"
    print_scores(model2, train_loader, val_loader, test_loader, title)


def third_model(train_loader_original):
    print("--------------------------- Model 3 --------------------------")
    # Third Architecture - Split:
    train_loader, val_loader = split_data(train_loader_original)

    model_3 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding='same'),
        # PrintLayer(layer_out=global_conv1_out),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # 14X14X32
        nn.Flatten(),
        nn.Linear(in_features=32 * 14 * 14, out_features=1024),
        nn.ReLU(),
        nn.Linear(in_features=1024, out_features=10)
    )

    model_3.apply(init_conv2d_weights)
    print(model_3[0].weight.data.shape)
    global activation
    activation = {}
    model_3[0].register_forward_hook(get_activation('conv1'))
    model_3[1].register_forward_hook(get_activation('conv1_Relu'))
    model_3[2].register_forward_hook(get_activation('conv1_maxpooled'))

    # Third Architecture - model training:
    model_training(model_3, train_loader, num_epochs=2, lr=1e-3, max_batches=13000)

    summary(model_3, (1, 28, 28))

    title = "Third Architecture - 3 : 1 Convolution Layer"
    print_scores(model_3, train_loader, val_loader, test_loader, title)

    return model_3, train_loader


def fourth_model(train_loader_original):
    print("--------------------------- Model 4 --------------------------")
    # Four Architecture - Split:
    train_loader, val_loader = split_data(train_loader_original)
    global activation
    activation = {}

    model_4 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding='same'),
        # PrintLayer(layer_out=global_conv1_out),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding='same'),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.Flatten(),
        nn.Linear(in_features=64 * 7 * 7, out_features=1024),
        nn.ReLU(),
        nn.Linear(in_features=1024, out_features=10)
    )

    model_4.apply(init_conv2d_weights)
    print(model_4[0].weight.data.shape)

    model_4[0].register_forward_hook(get_activation('conv1'))
    model_4[1].register_forward_hook(get_activation('conv1_Relu'))
    model_4[3].register_forward_hook(get_activation('conv2'))
    model_4[5].register_forward_hook(get_activation('conv2_maxpooled'))

    # Four Architecture - model training:
    model_training(model_4, train_loader, num_epochs=2, lr=1e-3, max_batches=13000)

    title = "Four Architecture - 4 : 2 Convolution Layer"
    print_scores(model_4, train_loader, val_loader, test_loader, title)

    summary(model_4, (1, 28, 28))

    return model_4, train_loader


def model_training_to_99(model, train_loader, val_loader, num_epochs=20, lr=0.001, max_batches=None):
    model.to(device)  # Move model to GPU/CPU
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    start_time = time.time()
    batch_count = 0  # Track number of batches processed
    iteration_count = 0  # Track total iterations across epochs

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}:")

        for batch_idx, (images, labels) in enumerate(train_loader):
            # Move images & labels to device
            images, labels = images.to(device), labels.to(device).long()

            batch_count += 1
            iteration_count += 1  # Count every training step

            # Stop training after `max_batches` if specified
            if max_batches and batch_count > max_batches:
                print(f"Reached {max_batches} batches, stopping batch processing for this epoch.")
                break

            # Forward pass
            predictions = model(images)
            loss = loss_fn(predictions, labels)

            # Print loss & accuracy every 100 batches
            if batch_idx % 100 == 0:
                acc = accuracy(predictions, labels).item()
                print(f"Batch {batch_idx}: CE = {loss.item():.6f}, ACC = {acc:.3f}")

            # Zero gradients
            optimizer.zero_grad()

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

        # Run validation after each epoch
        val_acc = validate_model(model, val_loader)
        print(f"Validation Accuracy: {val_acc:.2f}%")

        # Stop training if validation accuracy reaches 99%
        if val_acc >= 99.0:
            print(f"Achieved 99% validation accuracy in {iteration_count} iterations.")
            break

    # Final training metrics
    print(f"Final train step - CE: {loss.item():.6f}, ACC: {accuracy(predictions, labels).item():.3f}")

    # Total time taken
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    return iteration_count  # Return number of iterations needed


def validate_model(model, val_loader):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # No gradients needed for validation
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    model.train()  # Set back to training mode
    return (correct / total) * 100  # Return accuracy as percentage


def fivth_model(train_loader_original, device, minibatch=50):
    print(f"--------------------------- Model 5 - {minibatch} minibatch --------------------------")
    ##Split
    train_loader, val_loader = split_data(train_loader_original, minibatch=minibatch)
    global_conv1_out = 0
    global activation
    activation = {}  # Reset activation dictionary
    drop_out = 0.5
    model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding='same'),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding='same'),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.Flatten(),
        nn.Dropout(p=drop_out),
        nn.Linear(in_features=64 * 7 * 7, out_features=1024),
        nn.ReLU(),
        nn.Linear(in_features=1024, out_features=10)
    )
    model.apply(init_conv2d_weights)
    print(model[0].weight.data.shape)

    model[0].register_forward_hook(get_activation('conv1'))
    model[1].register_forward_hook(get_activation('conv1_Relu'))
    model[3].register_forward_hook(get_activation('conv2'))
    model[5].register_forward_hook(get_activation('conv2_maxpooled'))

    # model_training(model, train_loader, num_epochs=2, lr=1e-3, max_batches=13000)
    model_training_to_99(model, train_loader, val_loader , num_epochs=100, lr=0.001)

    title = "Five Architecture - 5 : 2 Convolution Layer + Dropout"
    print_scores(model, train_loader, val_loader, test_loader, title)

    summary(model, (1, 28, 28))

    # Fetch a batch from the dataset
    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)
    predictions = model(images)

    # Pick one image from the batch
    image_index = 3  # Choose a fixed index that will be used in all visualizations
    original_image = images[image_index].squeeze().cpu().numpy()
    label = labels[image_index].item()

    # Display the Original Image
    plt.figure(figsize=(6, 6))
    plt.title(f"Original Image - Label: {label}")
    plt.imshow(original_image) 
    plt.colorbar()
    plt.show()

    # Visualize Filter Weights (5 different filters from Conv1)**
    conv1_weights = model[image_index].weight.data.cpu().numpy()

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        axes[i].imshow(conv1_weights[i, 0])
        axes[i].set_title(f"Filter {i + 1}")
    plt.suptitle("Conv1 - Filter Weights")
    plt.show()

    feature_maps = {
        "Conv1": "conv1",
        "Conv1 ReLU": "conv1_Relu",
        "Conv2": "conv2",
        "Conv2 Maxpooled": "conv2_maxpooled"
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    for idx, (title, layer_name) in enumerate(feature_maps.items()):
        if layer_name in activation:
            filter_index = 4  # Choose a filter to visualize
            feature_map = activation[layer_name][image_index][filter_index].cpu()

            im = axes[idx].imshow(feature_map)
            plt.colorbar(im, ax=axes[idx]) 
            axes[idx].set_title(title)

    plt.show()

    return model, train_loader


def conv_image(model, image_batch, layer_num, channel_num, after_activation=True):
    model.eval()
    with torch.no_grad():  # No gradient calculation needed
        activations = image_batch  # Start with input images
        for i, layer in enumerate(model):  # Iterate through layers
            activations = layer(activations)
            if i == layer_num:  # Stop at the requested layer
                break

    if channel_num >= activations.shape[1]:  # Validate channel
        raise ValueError(f"Invalid channel_num {channel_num}. Max: {activations.shape[1] - 1}")

    feature_map = activations[0, channel_num].cpu().numpy()  
    plt.figure(figsize=(6, 6))
    plt.imshow(feature_map, cmap='viridis')  
    plt.colorbar()
    plt.title(
        f"Activation Map - Layer {layer_num}, Channel {channel_num} ({'After' if after_activation else 'Before'} Activation)")
    plt.show()


if __name__ == '__main__':
    device, train_loader_original, train_images, train_labels, test_loader = init()
    first_model(train_loader_original)
    second_model(train_loader_original)
    model_3,train_loader=third_model(train_loader_original)
    model_4, train_loader = fourth_model(train_loader_original)
    model_5_50, train_loader = fivth_model(train_loader_original, device, minibatch=50)
    model_5_100, train_loader = fivth_model(train_loader_original, device, minibatch=100)
    image_batch, _ = next(iter(train_loader))  # Fetch batch
    image_batch = image_batch.to(device)  # Move to device (CPU/GPU)
    conv_image(model_5_100, image_batch, layer_num=3, channel_num=3, after_activation=True)
