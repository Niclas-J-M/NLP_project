# train_sentiment_model.py

from sentimentmodel import SentimentModel

# Example dataset
texts = [
    "Ich liebe den neuen Film, er ist fantastisch!",
    "Das Wetter ist heute schrecklich.",
    "Dieses Produkt ist nur durchschnittlich.",
    "Das Konzert gestern Abend war erstaunlich.",
    "Ich bin kein Fan von diesem Restaurant.",
    "Neutrale Aussage über etwas.",
    "Der Service war wirklich schlecht.",
    "Das Buch war ziemlich angenehm.",
]

labels = ["positive", "negative", "neutral", "positive", "negative", "neutral", "negative", "positive"]

# Split the data into training and validation sets
from sklearn.model_selection import train_test_split

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Instantiate SentimentModel
model = SentimentModel()

# Fine-tune the model
model.fine_tune(train_texts, train_labels, val_texts, val_labels, output_dir="./fine_tuned_model")
