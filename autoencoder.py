import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Convert text to numerical representation (ASCII values)
def text_to_numbers(text):
    return np.array([ord(c) for c in text]) / 127  # Normalize ASCII values

# Convert numbers back to text
def numbers_to_text(numbers):
    return ''.join([chr(int(n * 127)) for n in numbers])

# Define input text
plaintext = "HELLO"
n_input = len(plaintext)  # Number of characters

data = text_to_numbers(plaintext).reshape(1, n_input)

# Define autoencoder model
encoding_dim = 3  # Smaller than input size to enforce compression

# Encoder
input_layer = keras.Input(shape=(n_input,))
encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)

# Decoder
decoded = layers.Dense(n_input, activation='sigmoid')(encoded)

autoencoder = keras.Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the model
autoencoder.fit(data, data, epochs=500, verbose=0)

# Encrypt (encode) and Decrypt (decode) the message
encrypted = autoencoder.predict(data)
decrypted = numbers_to_text(encrypted[0])

print("Original Text:", plaintext)
print("Decrypted Text:", decrypted)
