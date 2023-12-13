# train_sentiment_model.py
from sentimentmodel import SentimentModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from data_generation import generate_texts_labels

# Example dataset
texts, labels = generate_texts_labels(num_samples=500)

# Split the data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Instantiate SentimentModel without fine-tuning
model_without_fine_tuning = SentimentModel()

# Fine-tune the model
model_with_fine_tuning = SentimentModel()
model_with_fine_tuning.fine_tune(train_texts, train_labels, val_texts, val_labels, output_dir="./fine_tuned_model")

# Make predictions and calculate F1 scores without fine-tuning
val_predictions_without_fine_tuning = model_without_fine_tuning.predict_sentiment(val_texts)
val_f1_micro_without_fine_tuning = f1_score(val_labels, val_predictions_without_fine_tuning, average='micro')
print(f"Validation Micro-Average F1 Score (Without Fine-Tuning): {val_f1_micro_without_fine_tuning}")

# Make predictions and calculate F1 scores with fine-tuning
val_predictions_with_fine_tuning = model_with_fine_tuning.predict_sentiment(val_texts)
val_f1_micro_with_fine_tuning = f1_score(val_labels, val_predictions_with_fine_tuning, average='micro')
print(f"Validation Micro-Average F1 Score (With Fine-Tuning): {val_f1_micro_with_fine_tuning}")
