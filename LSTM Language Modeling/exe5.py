!pip install tensorflow-gpu
!pip install --upgrade tensorflow
!pip install keras-preprocessing-gpu
# %tensorflow_version 2.x
import tensorflow as tf
!pip install tensorboardX
# !pip install language_models

if tf.config.list_physical_devices('GPU'):
    device_name = tf.test.gpu_device_name()
else:
    device_name = "/CPU:0"  # Fallback to CPU

print(f"Using device: {device_name}")
# if device_name != '/device:GPU:0':
#   raise SystemError('GPU device not found')
# print('Found GPU at: {}'.format(device_name))
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# !git clone https://github.com/pbloem/language-models.git
!git clone https://github.com/GuyKabiri/language_models

import keras

import keras.backend as K
from keras.datasets import imdb
from keras.layers import  LSTM, Embedding, TimeDistributed, Input, Dense
from keras.models import Model
from tensorflow.python.client import device_lib

from tensorflow.keras.losses import sparse_categorical_crossentropy

from tqdm import tqdm
import os, random

from argparse import ArgumentParser

import numpy as np

from tensorboardX import SummaryWriter

from language_models import util

import importlib
import tensorflow as tf
import math

CHECK = 5

def generate_seq(model : Model, seed, size, temperature=1.0):
    """
    :param model: The complete RNN language model
    :param seed: The first few wordas of the sequence to start generating from
    :param size: The total size of the sequence to generate
    :param temperature: This controls how much we follow the probabilities provided by the network. For t=1.0 we just
        sample directly according to the probabilities. Lower temperatures make the high-probability words more likely
        (providing more likely, but slightly boring sentences) and higher temperatures make the lower probabilities more
        likely (resulting is weirder sentences). For temperature=0.0, the generation is _greedy_, i.e. the word with the
        highest probability is always chosen.
    :return: A list of integers representing a samples sentence
    """

    ls = seed.shape[0]

    # Due to the way Keras RNNs work, we feed the model a complete sequence each time. At first it's just the seed,
    # zero-padded to the right length. With each iteration we sample and set the next character.

    tokens = np.concatenate([seed, np.zeros(size - ls)])

    for i in range(ls, size):

        probs = model.predict(tokens[None,:])

        # Extract the i-th probability vector and sample an index from it
        next_token = util.sample_logits(probs[0, i-1, :], temperature=temperature)

        tokens[i] = next_token

    return [int(t) for t in tokens]

def sparse_loss(y_true, y_pred):
    return sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

class Args:
  epochs = 3 # Number of epochs
  embedding_size = 300 # Size of the word embeddings on the input layer.
  out_every = 1 # Output every n epochs.
  lr = 0.001 # Learning rate
  batch = 64 # Batch size from 128 to 64
  task = 'wikisimple'
  data = './data' # Data file. Should contain one sentence per line.
  lstm_capacity = 256
  max_length = 50 # Sentence max length. from None to 50
  top_words = 10000 # Word list size.
  limit = None # Character cap for the corpus - not relevant in our exercise.
  tb_dir = './runs/words' # Tensorboard directory
  seed = -1 # RNG seed. Negative for random (seed is printed for reproducability).
  extra = None # Number of extra LSTM layers.

options = Args()


importlib.reload(util)  # Force reload

"""Split the Data"""

# Load dataset using `util.load_words()`
dataset_path = util.DIR + '/datasets/wikisimple.txt'
data, w2i, i2w = util.load_words(dataset_path, vocab_size=options.top_words, limit=options.limit)

if data is None:
    raise ValueError("Error: `util.load_words()` returned None. Ensure the dataset is correctly loaded.")

# Convert data to NumPy array
data = np.array(data, dtype=object)

# Shuffle data to avoid ordering bias
np.random.seed(42)
np.random.shuffle(data)

# Split into Train (80%), Validation (10%), Test (10%)
total_size = len(data)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)

train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]

# Display dataset stats
dataset_stats = {
    "Total Sentences": total_size,
    "Train Sentences": len(train_data),
    "Validation Sentences": len(val_data),
    "Test Sentences": len(test_data),
    "Vocabulary Size": len(w2i),
}

# Save processed data for later steps
np.save("train_data.npy", train_data, allow_pickle=True)
np.save("val_data.npy", val_data, allow_pickle=True)
np.save("test_data.npy", test_data, allow_pickle=True)

# Print statistics
print("Dataset Split Statistics:")
for key, value in dataset_stats.items():
    print(f"{key}: {value}")



# Load dataset
train_data = np.load("train_data.npy", allow_pickle=True)
val_data = np.load("val_data.npy", allow_pickle=True)
test_data = np.load("test_data.npy", allow_pickle=True)

# Load word index
x, w2i, i2w = util.load_words(util.DIR + '/datasets/wikisimple.txt', vocab_size=options.top_words, limit=options.limit)
numwords = len(i2w)

# Pad sequences
x_max_len = max([len(sentence) for sentence in x])
x = util.batch_pad(x, options.batch, add_eos=True)

print(f"Loaded dataset: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

# Define function to create LSTM models
def create_lstm_model(num_layers=1, reverse=False):
    input_layer = Input(shape=(None,))
    embedding = Embedding(numwords, options.embedding_size, input_length=None)(input_layer)

    h = embedding
    for _ in range(num_layers):
        h = LSTM(options.lstm_capacity, return_sequences=True)(h)

    output_layer = TimeDistributed(Dense(numwords, activation='linear'))(h)

    model = Model(input_layer, output_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=options.lr),
                  loss=sparse_loss)
    return model


def calculate_perplexity(model, dataset, sample_size=5000):
    total_loss = 0.0
    total_words = 0

    dataset = list(dataset)  # Fix: Ensure dataset is a list before sampling
    dataset_sample = random.sample(dataset, min(len(dataset), sample_size))  # Select subset

    for batch in tqdm(dataset_sample):
        batch = np.array(batch, dtype=np.int32)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)

        n, l = batch.shape

        batch_shifted = np.concatenate([np.ones((n, 1), dtype=np.int32), batch], axis=1)
        batch_out = np.concatenate([batch, np.zeros((n, 1), dtype=np.int32)], axis=1)

        loss = model.evaluate(batch_shifted, batch_out[:, :, None], verbose=0)
        total_loss += loss * n
        total_words += n

    avg_loss = total_loss / total_words
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    return perplexity

# Define and train models
models = {
    "LSTM_1L_Normal": create_lstm_model(num_layers=1, reverse=False),
    "LSTM_2L_Normal": create_lstm_model(num_layers=2, reverse=False),
    "LSTM_1L_Reversed": create_lstm_model(num_layers=1, reverse=True),
    "LSTM_2L_Reversed": create_lstm_model(num_layers=2, reverse=True),
}

# Training loop for each model
perplexity_results = {}

for model_name, model in models.items():
    print(f"\n Training {model_name}...")

    tbw = SummaryWriter(log_dir=f"./runs/{model_name}")
    instances_seen = 0

    for epoch in range(options.epochs):
        for batch in tqdm(x):
            batch = np.array(batch, dtype=np.int32)  # Fix: Convert batch to int32
            n, l = batch.shape

            if "Reversed" in model_name:
                batch = np.flip(batch, axis=1)  # Reverse order

            batch_shifted = np.concatenate([np.ones((n, 1), dtype=np.int32), batch], axis=1)  # Prepend start symbol
            batch_out = np.concatenate([batch, np.zeros((n, 1), dtype=np.int32)], axis=1)  # Append pad symbol

            # Ensure batch_out is in integer format
            batch_out = batch_out.astype(np.int32)

            loss = model.train_on_batch(batch_shifted, batch_out[:, :, None])

            instances_seen += n
            tbw.add_scalar(f'lm/{model_name}-loss', float(loss), instances_seen)
        print(f"Epoch {epoch + 1}, Loss: {loss}")

    # Save model in recommended `.keras` format
    tf.keras.models.save_model(model, f"{model_name}.keras")  # Fix: Removed `keras.saving`
    print(f"{model_name} training complete. Model saved.")

    # Compute Perplexity
    print(f"Calculating perplexity for {model_name}...")
    perplexity_results[model_name] = {
        "Train Perplexity": calculate_perplexity(model, train_data),
        "Validation Perplexity": calculate_perplexity(model, val_data),
        "Test Perplexity": calculate_perplexity(model, test_data),
    }
    print(f"Perplexity for {model_name}: {perplexity_results[model_name]}")

# Print Final Perplexity Results
print("\nFinal Perplexity Results:")
for model_name, results in perplexity_results.items():
    print(f"\n {model_name}")
    for dataset, ppl in results.items():
        print(f"{dataset}: {ppl:.4f}")

def generate_sentences_with_temperature(model, seed_text, max_length=7):
    """
    Generates sentences using different temperatures.

    :param model: Trained LSTM model.
    :param seed_text: The initial words (e.g., "I love").
    :param max_length: The total number of words in the generated sentence.
    """
    # Convert words to indices
    seed_sequence = [w2i[word] if word in w2i else w2i["<UNK>"] for word in seed_text.split()]
    seed_sequence = np.array(seed_sequence)  # Convert to numpy array

    print(f"\nGenerating text for seed: '{seed_text}'")

    for temp in [0.1, 1, 10]:
        # Generate sequence
        generated_indices = generate_seq(model, seed_sequence, max_length, temperature=temp)

        # Convert indices back to words
        generated_sentence = ' '.join(i2w[idx] for idx in generated_indices)

        print(f"\nTemperature = {temp}")
        print(f"Generated: {generated_sentence}")

# Run the function
generate_sentences_with_temperature(model, "I love")


custom_objects = {"sparse_loss": sparse_loss}
model_1L_Normal = tf.keras.models.load_model("LSTM_1L_Normal.keras", custom_objects=custom_objects)
model_1L_Reversed = tf.keras.models.load_model("LSTM_1L_Reversed.keras", custom_objects=custom_objects)
model_2L_Normal = tf.keras.models.load_model("LSTM_2L_Normal.keras", custom_objects=custom_objects)
model_2L_Reversed = tf.keras.models.load_model("LSTM_2L_Reversed.keras", custom_objects=custom_objects)

print("All models loaded successfully!")

model_1L_Normal.summary()
model_1L_Reversed.summary()
model_2L_Normal.summary()
model_2L_Reversed.summary()

def sentence_probability_from_text(model, sentence, w2i):
    """
    Compute the probability of a sentence given a trained LSTM language model.

    :param model: Trained LSTM model (Keras).
    :param sentence: String input (actual sentence).
    :param w2i: Word-to-index dictionary from util.load_words().
    :return: Probability of the sentence under the model.
    """

    # Tokenize sentence using w2i (word-to-index mapping)
    sentence_tokens = [w2i[word] if word in w2i else w2i["<UNK>"] for word in sentence.split()]

    if not sentence_tokens:
        raise ValueError("Sentence contains no valid words from vocabulary.")

    sentence_tokens = np.array(sentence_tokens, dtype=np.int32)

    # Add start token at the beginning
    sentence_tokens = np.insert(sentence_tokens, 0, 1)  # Assuming token '1' is the start token

    n = sentence_tokens.shape[0]

    # Prepare input for model
    tokens = np.zeros((1, n), dtype=np.int32)
    tokens[0, :n] = sentence_tokens

    # Get model predictions
    probs = model.predict(tokens, verbose=0)[0]

    # Compute sentence probability
    total_log_prob = 0.0

    for i in range(1, n):  # Skip the first token because there's no prediction before it
        word_index = sentence_tokens[i]
        word_prob = tf.nn.softmax(probs[i - 1])  # Get probability distribution for the previous step
        prob = word_prob[word_index].numpy()  # Extract probability of actual word

        if prob > 0:
            total_log_prob += np.log(prob)  # Sum log probabilities to prevent underflow
        else:
            total_log_prob += np.log(1e-10)  # Avoid log(0) errors

    sentence_probability = np.exp(total_log_prob)  # Convert back from log-probability

    return sentence_probability

x, w2i, i2w = util.load_words(util.DIR + '/datasets/wikisimple.txt', vocab_size=options.top_words, limit=options.limit)

sentence = "hello world"

#Use one of your trained models
prob = sentence_probability_from_text(model_1L_Normal, sentence, w2i)
print(f"The sentence: {sentence} \nHas probability of: {prob:.8f}")

def predict_next_word(model, user_input, w2i, i2w, temperature=1.0):
    """
    Predict the next word based on user input using the trained LSTM model.

    :param model: Trained LSTM model (Keras).
    :param user_input: A single word or phrase provided by the user.
    :param w2i: Word-to-index dictionary.
    :param i2w: Index-to-word list.
    :param temperature: Controls randomness in sampling (1.0 = normal, <1.0 = conservative, >1.0 = creative).
    :return: Predicted next word.
    """

    # Tokenize input sentence
    sentence_tokens = [w2i[word] if word in w2i else w2i.get("<UNK>", 0) for word in user_input.split()]

    if not sentence_tokens:
        raise ValueError("Input sentence contains no known words from the vocabulary.")

    sentence_tokens = np.array(sentence_tokens, dtype=np.int32)

    # Add start token if needed
    sentence_tokens = np.insert(sentence_tokens, 0, w2i.get("<START>", 1))  # Assuming '<START>' token exists

    n = sentence_tokens.shape[0]

    # Prepare input for model
    tokens = np.zeros((1, n), dtype=np.int32)
    tokens[0, :n] = sentence_tokens

    # Get model prediction
    preds = model.predict(tokens, verbose=0)[0, -1, :]  # Get last word prediction

    # Prevent log(0) errors
    preds = np.clip(preds, 1e-10, None)

    # Apply temperature scaling
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    probabilities = exp_preds / np.sum(exp_preds)  # Normalize probabilities

    # Sample the next word
    next_word_id = np.random.choice(len(probabilities), p=probabilities)

    # Convert token ID back to word using list indexing
    next_word = i2w[next_word_id] if next_word_id < len(i2w) else "<UNK>"

    return next_word

user_input = input("Enter a word: ")

#Predict the next word using one of your trained models
next_word = predict_next_word(model_1L_Normal, user_input, w2i, i2w)

print(f"Predicted next word: {next_word}")

sentence1 = "I love 1873 Derbyshire movements Schoenberg meter"
sentence2 = "I love Royal the . recently state"
sentence3 = "I love post Grange ancestor Colonel Alonso"
sentence4 = "i love cupcakes"

#Use one of your trained models
prob = sentence_probability_from_text(model_1L_Reversed, sentence1, w2i)
print(f"The sentence: {sentence1} \nHas probability of: {prob:.8f}")
prob = sentence_probability_from_text(model_1L_Reversed, sentence2, w2i)
print(f"The sentence: {sentence2} \nHas probability of: {prob:.8f}")
prob = sentence_probability_from_text(model_1L_Reversed, sentence3, w2i)
print(f"The sentence: {sentence3} \nHas probability of: {prob:.8f}")
prob = sentence_probability_from_text(model_1L_Reversed, sentence4, w2i)
print(f"The sentence: {sentence4} \nHas probability of: {prob:.8f}")