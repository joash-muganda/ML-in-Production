# Databricks notebook source
# MAGIC %pip install tensorflow tensorflow-datasets

# COMMAND ----------

import tensorflow_datasets as tfds

try:
    # Load dataset
    ds, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True, split='train')

    # Iterate through the dataset
    for i, (text, label) in enumerate(ds):
        print(f"Review {i+1}: {text.numpy()[:50]}... Label: {'Positive' if label else 'Negative'}")
        if i == 10:  # Limit to first 10 reviews to keep output manageable
            break

except Exception as e:
    print(f"An error occurred: {e}")


# COMMAND ----------

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load dataset
ds, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True, split='train', batch_size=-1)
train_data = tfds.as_numpy(ds)

# Unpack the reviews and labels
train_texts, train_labels = train_data

# Decode the texts
train_texts = [text.decode('utf-8') for text in train_texts]

# Initialize tokenizer
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)

# Convert texts to sequences
train_sequences = tokenizer.texts_to_sequences(train_texts)

# Padding sequences to ensure uniform input size
train_padded = pad_sequences(train_sequences, maxlen=120, truncating='post', padding='post')

print("Example of padded sequences:", train_padded[0])


# COMMAND ----------

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
train_padded, test_padded, train_labels, test_labels = train_test_split(train_padded, train_labels, test_size=0.2, random_state=42)


# COMMAND ----------

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
train_padded, test_padded, train_labels, test_labels = train_test_split(train_padded, train_labels, test_size=0.2, random_state=42)


# COMMAND ----------

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import mlflow.tensorflow

# Enable automatic logging with MLflow
mlflow.tensorflow.autolog()

# Model architecture adjustments
model = Sequential([
    Embedding(input_dim=10000, output_dim=16, input_length=120),
    Bidirectional(LSTM(32, return_sequences=True)),  # Reduced LSTM units
    Dropout(0.6),  # Increased Dropout rate
    Bidirectional(LSTM(16)),  # Reduced LSTM units in the second layer
    Dropout(0.4),  # Increased Dropout rate
    Dense(1, activation='sigmoid')
])

# Optimizer adjustment
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)  # Further reduced learning rate

# Compile the model
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping implementation
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Model summary
model.summary()

# Training the model with early stopping
history = model.fit(
    train_padded, train_labels,
    epochs=30,  # Increased number of epochs
    validation_data=(test_padded, test_labels),
    callbacks=[early_stopping],
    batch_size=64  # Adjusted batch size
)





# COMMAND ----------


train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# COMMAND ----------

import matplotlib.pyplot as plt

# Set up the figure size and labels
plt.figure(figsize=(12, 5))

# Plotting training and validation loss
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

