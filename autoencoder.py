import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define character set (A-Z, a-z, 0-9, space, punctuation)
chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?;:'\""
char_to_int = {char: i for i, char in enumerate(chars)}
int_to_char = {i: char for char, i in char_to_int.items()}
num_classes = len(chars)

# Define maximum sentence length
MAX_LEN = 50  # Change as needed

# Function to pad text to MAX_LEN
def pad_text(text, max_len):
    return text.ljust(max_len)[:max_len]  # Pads with spaces or trims if too long

# Function to one-hot encode text
def one_hot_encode(text, num_classes, max_len):
    encoded = np.zeros((max_len, num_classes))
    for i, char in enumerate(text):
        if char in char_to_int:
            encoded[i, char_to_int[char]] = 1
    return encoded

# Function to decode one-hot encoded text
def decode_text(encoded_output):
    decoded_chars = [int_to_char[np.argmax(char)] for char in encoded_output]
    return "".join(decoded_chars).strip()  # Remove padding spaces

# Build Autoencoder Model
input_layer = keras.Input(shape=(num_classes,))
encoded = layers.Dense(10, activation="relu")(input_layer)  # Compression
decoded = layers.Dense(num_classes, activation="softmax")(encoded)  # Reconstruction

autoencoder = keras.Model(input_layer, decoded)
autoencoder.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Function to train autoencoder on input text
def train_autoencoder(text):
    text = pad_text(text, MAX_LEN)
    data = one_hot_encode(text, num_classes, MAX_LEN)
    autoencoder.fit(data, data, epochs=5000, verbose=0)  # Train to memorize the text

# Function to encode & decode text
def encrypt_decrypt(text):
    text = pad_text(text, MAX_LEN)
    data = one_hot_encode(text, num_classes, MAX_LEN)
    
    # Encrypt
    encoded_text = autoencoder.layers[1](data).numpy()
    
    # Decrypt
    decoded_text = autoencoder.predict(data)
    decrypted_text = decode_text(decoded_text)
    
    return encoded_text, decrypted_text

# Take user input
user_input = input("Enter a sentence to encrypt & decrypt: ")
train_autoencoder(user_input)  # Train the model on user input

# Encrypt & Decrypt
encoded, decrypted = encrypt_decrypt(user_input)

# Display results
print("\nOriginal Text: ", user_input)
print("Encoded Representation:\n", encoded)
print("Decrypted Text: ", decrypted)
