"""
•	Create a LSTM  model for text analysis using Python and TensorFlow.
o	(Input corpus = “Anna, embrace change”
o	“Change is  constant”.
o	“She is positive thinker”
o	“Anna takes life as a challenge”)
"""

import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.optimizers import Adam

# Sample larger input text corpus
corpus = [
    "Anna, embrace change",
    "Change is constant",
    "She is positive thinke",
    "Anna takes life as a challenge"
]

# Prepare tokenizer and word index
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
word_index = tokenizer.word_index
total_words = len(word_index) + 1

# Prepare input-output pairs
sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        sequences.append(n_gram_sequence)

# Pad sequences and create predictors and label
max_sequence_len = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_sequence_len, padding='pre')
x_train, y_train = sequences[:, :-1], sequences[:, -1]

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, num_classes=total_words)

# Build LSTM model
model = Sequential()
model.add(Embedding(total_words, 50, input_length=max_sequence_len - 1))  # Embedding layer with 50 dimensions
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01))

# Train the model
model.fit(x_train, y_train, epochs=500, verbose=1)


# Function to generate text with temperature
def generate_text(seed_text, next_words, model, tokenizer, max_sequence_len, temperature=1.0):
    output_text = seed_text
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([output_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)[0]
        predicted_probs = np.asarray(predicted_probs).astype('float64')
        predicted_probs = np.log(predicted_probs) / temperature
        exp_preds = np.exp(predicted_probs)
        predicted_probs = exp_preds / np.sum(exp_preds)
        predicted_word_index = np.random.choice(range(len(predicted_probs)), p=predicted_probs)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                output_word = word
                break
        output_text += " " + output_word
    return output_text


# Function to generate multiple sentences
def generate_sentences(seed_text, num_sentences, words_per_sentence, model, tokenizer, max_sequence_len,
                       temperature=1.0):
    sentences = []
    for _ in range(num_sentences):
        sentence = generate_text(seed_text, words_per_sentence, model, tokenizer, max_sequence_len, temperature)
        sentences.append(sentence.strip())
    return ', '.join(sentences)


# Generate multiple sentences
seed_text = "Anna"
num_sentences = 5
words_per_sentence = 10
temperature = 1.0  # Adjust temperature for more or less randomness
generated_text = generate_sentences(seed_text, num_sentences, words_per_sentence, model, tokenizer, max_sequence_len,
                                    temperature)
print(generated_text)

