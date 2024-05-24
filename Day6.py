#. Use Keras library to create and train a basic RNN for  sequence prediction. Use a  simple sequence (8, 9, 10,11,12,13) and tries to predict the next number in the sequence.

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

# Define the sequence
sequence = np.array([8, 9, 10, 11, 12, 13])

# Create input sequences and corresponding targets
X = sequence[:-3]
y = sequence[3:]

# Build the RNN model
model = Sequential()
#model.add(SimpleRNN(units=1, activation='relu', input_shape=(None, 1)))
model.add(Dense(units=1,input_shape=[1]))

# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=10, verbose=0)

# Predict the next number
next_number = model.predict([13])
print(f"Predicted next number: {next_number}")

#2.Create a Basic Recurrent Neural Network (RNN)  model for text analysis using Python and TensorFlow.(Input text=”Anna, embrace change; Change is  constant.”)

import numpy as np
import tensorflow as tf

# Sample text data
text="Anna, embrace change; Change is  constant."

# Create a vocabulary
vocab = sorted(set(text))

# Create mappings from characters to indices and vice versa
char_to_idx = {char: idx for idx, char in enumerate(vocab)}
idx_to_char = np.array(vocab)

# Convert text to numerical data
text_as_int = np.array([char_to_idx[char] for char in text])

# Reshape input data to include batches of sequences
input_text = text_as_int[:-1]
target_text = text_as_int[1:]
input_text = np.expand_dims(input_text, axis=0)
target_text = np.expand_dims(target_text, axis=0)


# Define the RNN model
embedding_dim = 256
rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.SimpleRNN(rnn_units, return_sequences=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

model = build_model(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=1
)

# Compile the model
model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True))

# Define the number of training epochs
num_epochs = 100

# Train the model
model.fit(input_text, target_text, epochs=num_epochs)

# Generate text
def generate_text(model, start_string, num_generate=1000):
    input_eval = [char_to_idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []

    model.reset_states()
    for _ in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx_to_char[predicted_id])

    return start_string + ''.join(text_generated)

# Generate text
generated_text = generate_text(model, start_string="Anna", num_generate=1000)
print(generated_text)

