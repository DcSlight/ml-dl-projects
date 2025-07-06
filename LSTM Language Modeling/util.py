
from keras.layers import Layer
import keras.utils
import keras.backend as K

from nltk import FreqDist
import numpy as np

from keras.preprocessing.sequence import pad_sequences

from scipy.special import logsumexp

from collections import defaultdict, Counter, OrderedDict

import os
import re

def text_to_word_sequence(text):
   text = text.lower() # Convert to lowercase
   text = re.sub(r'[^\w\s]', '', text) # Remove punctuation return text.split() # Split into words

"""
Various utility functions for loading data and performing other common operations.

Some of this code is based on Based on https://github.com/ChunML/seq2seq/blob/master/seq2seq_utils.py
"""


# Special tokens
EXTRA_SYMBOLS = ['<PAD>', '<START>', '<UNK>', '<EOS>']
DIR = os.path.dirname(os.path.realpath(__file__))

def load_words(source, vocab_size=10000, limit=None, max_length=None):
    """
    Loads text data, tokenizes it, and returns a list of word indices.

    :param source: Path to the dataset file.
    :param vocab_size: Maximum vocabulary size.
    :param limit: Maximum number of characters to read from the file.
    :param max_length: Maximum sentence length.
    :return: Tuple (processed sentences, word-to-index mapping, index-to-word mapping)
    """

    print(f"Loading file: {source}")

    # Check if file exists
    if not os.path.exists(source):
        print(f"Error: File '{source}' not found!")
        return None

    # Try opening the file and reading contents
    try:
        with open(source, 'r', encoding='utf-8') as f:
            text = f.read(limit) if limit else f.read()

        # Split into sentences
        lines = text.split("\n")

        print(f"Read {len(lines)} lines from the file.")

        if not lines or len(lines) == 1 and lines[0] == "":
            print("Error: File is empty or not properly formatted.")
            return None

        # Tokenize sentences into words
        x = []
        for i, sentence in enumerate(lines):
            words = sentence.strip().split()
            if words:
                x.append(words)

        if not x:
            print("Error: No valid sentences found.")
            return None

        print(f"Processed {len(x)} valid sentences.")

        # Build vocabulary (word-to-index mapping)
        word_freq = {}
        for sentence in x:
            for word in sentence:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency and keep only top `vocab_size` words
        sorted_vocab = sorted(word_freq.items(), key=lambda item: item[1], reverse=True)[:vocab_size]
        i2w = ["<PAD>", "<UNK>"] + [word for word, _ in sorted_vocab]
        w2i = {word: i for i, word in enumerate(i2w)}

        print(f"Vocabulary size: {len(i2w)} words.")

        # Convert words to indices
        x_indexed = [[w2i.get(word, w2i["<UNK>"]) for word in sentence] for sentence in x]

        return x_indexed, w2i, i2w

    except Exception as e:
        print(f"Error while processing file: {e}")
        return None

def load_characters(source, length=None, limit=None,):
    """
    Reads a text file as a stream of characters. The stream is cut into chunks of equal size

    :param source: The text file to read
    :param length: The size of the chunks. If None, the stream is delimited by line-ends and the resulting sequence will
        have variable length
    :param limit: If not None, only the first "character_limit" characters are read. Useful for debugging on large corpora.
    :return: (1) A list of lists
    """

    # Reading raw text from source and destination files
    f = open(source, 'r')
    x_data = f.read()
    f.close()

    print('raw data read')

    if limit is not None:
        x_data = x_data[:limit]

    # Splitting raw text into array of sequences
    if length is None:
        x = [list(line) for line in x_data.split('\n') if len(line) > 0]
    else:
        x = [list(chunk) for chunk in chunks(x_data, length)]

    # Creating the vocabulary set with the most common characters (leaving room for PAD, START, UNK)
    chars = set()
    for line in x:
        for char in line:
            chars.add(char)

    # Creating an array of words from the vocabulary set, we will use this array as index-to-word dictionary
    i2c = list(chars)
    # Adding the special symbol to the beginning of the array
    i2c = EXTRA_SYMBOLS + i2c

    # Creating the word-to-index dictionary from the array created above
    c2i = {word:ix for ix, word in enumerate(i2c)}

    # Converting each word to its index value
    for i, sentence in enumerate(x):
        for j, word in enumerate(sentence):
            if word in c2i:
                x[i][j] = c2i[word]
            else:
                x[i][j] = c2i['<UNK>']

    return x, c2i, i2c

def process_data(word_sentences, max_len, word_to_ix):
    # Vectorizing each element in each sequence
    sequences = np.zeros((len(word_sentences), max_len, len(word_to_ix)))
    for i, sentence in enumerate(word_sentences):
        for j, word in enumerate(sentence):
            sequences[i, j, word] = 1.
    return sequences


def batch_pad(x, batch_size, min_length=3, add_eos=False, extra_padding=0):
    """
    Takes a list of integer sequences, sorts them by lengths and pads them so that sentences in each batch have the
    same length.

    :param x:
    :return: A list of tensors containing equal-length sequences padded to the length of the longest sequence in the batch
    """

    x = sorted(x, key=lambda l : len(l))

    if add_eos:
        eos = EXTRA_SYMBOLS.index('<EOS>')
        x = [sent + [eos,] for sent in x]

    batches = []

    start = 0
    while start < len(x):
        end = start + batch_size
        if end > len(x):
            end = len(x)

        batch = x[start:end]

        mlen = max([len(l) + extra_padding for l in batch])

        if mlen >= min_length:
            batch = pad_sequences(batch, maxlen=mlen, dtype='int32', padding='post', truncating='post')

            batches.append(batch)

        start += batch_size


    print('max length per batch: ', [max([len(l) for l in batch]) for batch in batches])
    return batches

def to_categorical(batch, num_classes):
    """
    Converts a batch of length-padded integer sequences to a one-hot encoded sequence
    :param batch:
    :param num_classes:
    :return:
    """

    b, l = batch.shape

    out = np.zeros((b, l, num_classes))

    for i in range(b):
        seq = batch[0, :]
        out[i, :, :] = keras.utils.to_categorical(seq, num_classes=num_classes)

    return out

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def sample(preds, temperature=1.0):
    """
    Sample an index from a probability vector

    :param preds:
    :param temperature:
    :return:
    """

    preds = np.asarray(preds).astype('float64')

    if temperature == 0.0:
        return np.argmax(preds)

    preds = np.log(preds) / temperature

    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    probas = np.random.multinomial(1, preds, 1)

    return np.argmax(probas)

def sample_logits(preds, temperature=1.0):
    """
    Sample an index from a logit vector.

    :param preds:
    :param temperature:
    :return:
    """
    preds = np.asarray(preds).astype('float64')

    if temperature == 0.0:
        return np.argmax(preds)

    preds = preds / temperature
    preds = preds - logsumexp(preds)

    choice = np.random.choice(len(preds), 1, p=np.exp(preds))

    return choice

class KLLayer(Layer):

    """
    Identity transform layer that adds KL divergence
    to the final model loss.

    During training, call
            K.set_value(kl_layer.weight, new_value)
    to scale the KL loss term.

    based on:
    http://tiao.io/posts/implementing-variational-autoencoders-in-keras-beyond-the-quickstart-tutorial/
    """

    def __init__(self, weight = None, *args, **kwargs):
        self.is_placeholder = True
        self.weight = weight
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        loss =  K.mean(kl_batch)
        if self.weight is not None:
            loss = loss * self.weight

        self.add_loss(loss, inputs=inputs)

        return inputs

class Sample(Layer):
    """
    Performs sampling step
    """
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super().__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var, eps = inputs

        z = K.exp(.5 * log_var) * eps + mu

        return z

    def compute_output_shape(self, input_shape):
        shape_mu, _, _ = input_shape
        return shape_mu

def interpolate(start, end, steps):

    result = np.zeros((steps+2, start.shape[0]))
    for i, d in enumerate(np.linspace(0,1, steps+2)):
        result[i, :] = start * (1-d) + end * d

    return result


class OrderedCounter(Counter, OrderedDict):
    'Counter that remembers the order elements are first encountered'

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)

def idx2word(idx, i2w, pad_idx):

    sent_str = [str()]*len(idx)

    for i, sent in enumerate(idx):

        for word_id in sent:

            if word_id == pad_idx:
                break
            sent_str[i] += i2w[str(word_id.item())] + " "

        sent_str[i] = sent_str[i].strip()


    return sent_str
