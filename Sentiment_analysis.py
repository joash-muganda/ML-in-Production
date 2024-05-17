# Databricks notebook source
# MAGIC %pip install tensorflow tensorflow-datasets

# COMMAND ----------

# MAGIC %pip install scikit-learn
# MAGIC

# COMMAND ----------

# MAGIC %pip install nltk
# MAGIC

# COMMAND ----------

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')


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

# import tensorflow_datasets as tfds
# import nltk
# from nltk.sentiment import SentimentIntensityAnalyzer
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# # Ensure required NLTK resources are downloaded
# nltk.download('vader_lexicon')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

# try:
#     # Load dataset only once
#     ds, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True, split='train')

#     # Iterate through the dataset for initial exploration
#     for i, (text, label) in enumerate(ds.take(10)):  # Example: Only taking the first 10 for brevity
#         print(f"Review {i+1}: {text.numpy()[:50]}... Label: {'Positive' if label.numpy() == 1 else 'Negative'}")

# except Exception as e:
#     print(f"An error occurred: {e}")

# # Assuming the same session continues and ds is still in scope

# # Initialize sentiment analysis and tokenizer
# sia = SentimentIntensityAnalyzer()
# tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")

# def preprocess_and_extract_features(ds_subset):
#     processed_texts = []
#     labels = []
#     features = []

#     # Process each entry in the dataset subset
#     for text, label in ds_subset:
#         decoded_text = text.numpy().decode('utf-8')
#         processed_texts.append(decoded_text)
#         labels.append(label.numpy())

#         # Perform sentiment analysis
#         sentiment_score = sia.polarity_scores(decoded_text)
#         features.append({
#             'sentiment_neg': sentiment_score['neg'],
#             'sentiment_neu': sentiment_score['neu'],
#             'sentiment_pos': sentiment_score['pos'],
#             'sentiment_compound': sentiment_score['compound']
#         })

#     return processed_texts, labels, features

# # Example of further processing using the same dataset
# # Here, we assume further processing of a different subset or the same as needed
# texts, labels, sentiment_features = preprocess_and_extract_features(ds.take(100))

# # Tokenize and pad sequences
# tokenizer.fit_on_texts(texts)
# sequences = tokenizer.texts_to_sequences(texts)
# padded_sequences = pad_sequences(sequences, maxlen=120, truncating='post', padding='post')

# # Example output to verify correct processing
# print("Example of padded sequences:", padded_sequences[0])

import tensorflow_datasets as tfds
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Ensure required NLTK resources are downloaded
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

try:
    # Load dataset only once
    ds, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True, split='train')
    
    # Initialize sentiment analysis and tokenizer
    sia = SentimentIntensityAnalyzer()
    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    
    processed_texts = []
    labels = []
    features = []
    
    # Process each entry in the dataset
    for text, label in ds.take(100):  # Processing more data for feature storage
        decoded_text = text.numpy().decode('utf-8')
        processed_texts.append(decoded_text)
        labels.append(label.numpy())
        
        # Perform sentiment analysis
        sentiment_score = sia.polarity_scores(decoded_text)
        features.append({
            'sentiment_neg': sentiment_score['neg'],
            'sentiment_neu': sentiment_score['neu'],
            'sentiment_pos': sentiment_score['pos'],
            'sentiment_compound': sentiment_score['compound']
        })
    
    # Tokenize and pad sequences
    tokenizer.fit_on_texts(processed_texts)
    sequences = tokenizer.texts_to_sequences(processed_texts)
    padded_sequences = pad_sequences(sequences, maxlen=120, truncating='post', padding='post')

    # Prepare DataFrame or structured data for feature store
    # Include labels and padded sequences for training
    # Example DataFrame creation (ensure you have pandas installed):
    import pandas as pd
    df = pd.DataFrame(features)
    df['labels'] = labels
    df['padded_sequences'] = list(padded_sequences)  # Store sequences in a way compatible with your DataFrame handling
    
    print("Data ready for feature store and training:", df.head())

except Exception as e:
    print(f"An error occurred: {e}")


# COMMAND ----------

# Convert the pandas DataFrame to a Spark DataFrame
spark_features_df = spark.createDataFrame(df)


# COMMAND ----------


spark_features_df.show()


# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id
from databricks.feature_store import FeatureStoreClient

# Start or get the current Spark session
spark = SparkSession.builder.appName("Feature Store Setup").getOrCreate()

# Assuming 'spark_features_df' is already your Spark DataFrame
spark_features_df = spark_features_df.withColumn("review_id", monotonically_increasing_id())

# Initialize the Feature Store Client
fs = FeatureStoreClient()

# Drop the existing feature table
fs.drop_table(name="default.sentiment_features")  # Ensure 'default' is your target database

# Create a new feature table
fs.create_table(
    name="default.sentiment_features",
    primary_keys="review_id",
    df=spark_features_df,
    description="Features extracted for sentiment analysis",
)



# COMMAND ----------

# Access the feature table
from databricks.feature_store import FeatureStoreClient
fs = FeatureStoreClient()

# Read the feature table
features_df = fs.read_table("ankoro_akia.default.sentiment_features")

# Display the first few records to confirm
display(features_df)


# COMMAND ----------

# MAGIC %md
# MAGIC Training

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient
import numpy as np

# Initialize the Feature Store Client
fs = FeatureStoreClient()

# Read the feature table
features_df = fs.read_table("default.sentiment_features")

# Convert Spark DataFrame to Pandas DataFrame for ease of manipulation
features_pd = features_df.toPandas()

# Extract features and labels
X = np.array(features_pd[['sentiment_neg', 'sentiment_neu', 'sentiment_pos', 'sentiment_compound']])
y = features_pd['labels'].values

# Display basic information to verify correct loading
print("Features shape:", X.shape)
print("Labels shape:", y.shape)


# COMMAND ----------

# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout

# # Define the model
# model = Sequential([
#     Dense(16, activation='relu', input_shape=(4,)),  # Input layer with 4 features
#     Dropout(0.5),
#     Dense(16, activation='relu'),  # Hidden layer
#     Dropout(0.5),
#     Dense(1, activation='sigmoid')  # Output layer for binary classification
# ])

# # Compile the model
# model.compile(optimizer='adam', 
#               loss='binary_crossentropy', 
#               metrics=['accuracy'])

# # Display the model's architecture
# model.summary()


# COMMAND ----------

# from sklearn.model_selection import train_test_split

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train the model
# history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), batch_size=25)

# # Plot training history
# import matplotlib.pyplot as plt

# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()

# plt.show()


# COMMAND ----------

# from sklearn.metrics import classification_report, confusion_matrix

# # Predict classes with the test set
# y_pred = model.predict(X_test)
# y_pred_classes = (y_pred > 0.5).astype(int)  # Threshold to convert probabilities to binary class output

# # Detailed classification report
# print(classification_report(y_test, y_pred_classes))

# # Confusion matrix
# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred_classes))


# COMMAND ----------

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Define the model with a slightly lower dropout rate to potentially increase learning capacity
model = Sequential([
    Dense(16, activation='relu', input_shape=(4,)),  # Input layer with 4 features
    Dropout(0.3),  # Reduced dropout rate
    Dense(16, activation='relu'),  # Hidden layer
    Dropout(0.3),  # Reduced dropout rate
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Configure the optimizer with Nesterov momentum
optimizer = SGD(learning_rate=0.001, momentum=0.9, nesterov=True)

# Compile the model
model.compile(optimizer=optimizer, 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Display the model's architecture
model.summary()


# COMMAND ----------

from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_no = 1
losses, accuracies, val_losses, val_accuracies = [], [], [], []

# Define different batch sizes for each fold
batch_sizes = [10, 15, 20, 25, 30]

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    batch_size = batch_sizes[fold_no - 1]  # Dynamic batch size selection

    # Train the model
    print(f'Training on fold {fold_no} with batch size {batch_size}...')
    history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), batch_size=batch_size)

    # Save the history from each fold
    losses.extend(history.history['loss'])
    accuracies.extend(history.history['accuracy'])
    val_losses.extend(history.history['val_loss'])
    val_accuracies.extend(history.history['val_accuracy'])
    fold_no += 1

# Plotting aggregate training history across all folds
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Aggregate Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Aggregate Training and Validation Accuracy')
plt.legend()

plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Registering Model

# COMMAND ----------

# MAGIC %pip install "mlflow-skinny[databricks]>=2.4.1"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow

# Set the registry URI to point to Databricks' model registry
mlflow.set_registry_uri("databricks-uc")

# Register the model directly without specifying catalog or schema
model_name = "sentiment_analysis_model"  # Ensure this model name is unique in your workspace
model_uri = "runs:/b139b254ac144ce3ba7a5b87578fb2a2/model"
mlflow.register_model(model_uri, model_name)

