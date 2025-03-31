import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define character set (HELLO only)
chars = sorted(set("HELLO"))  # Unique characters: H, E, L, O
char_to_int = {char: i for i, char in enumerate(chars)}
int_to_char = {i: char for char, i in char_to_int.items()}

# One-hot encode "HELLO"
def one_hot_encode(text, num_classes):
    encoded = np.zeros((len(text), num_classes))
    for i, char in enumerate(text):
        encoded[i, char_to_int[char]] = 1
    return encoded

# One-hot encoded input
num_classes = len(chars)
data = one_hot_encode("HELLO", num_classes)

# Define autoencoder model
input_layer = keras.Input(shape=(num_classes,))
encoded = layers.Dense(3, activation="relu")(input_layer)  # Compression
decoded = layers.Dense(num_classes, activation="softmax")(encoded)  # Reconstruction

autoencoder = keras.Model(input_layer, decoded)

# Compile model
autoencoder.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
autoencoder.fit(data, data, epochs=5000, verbose=0)

# Encode and decode the input
encoded_text = autoencoder.layers[1](data).numpy()
decoded_text = autoencoder.predict(data)

# Convert back to characters
decoded_chars = [int_to_char[np.argmax(c)] for c in decoded_text]

# Print results
print("Original Text: ", "HELLO")
print("Encoded Representation:\n", encoded_text)
print("Decoded Text: ", "".join(decoded_chars))
