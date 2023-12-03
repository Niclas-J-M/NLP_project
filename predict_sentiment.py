# predict_sentiment.py

from sentimentmodel import SentimentModel

# Instantiate SentimentModel
model = SentimentModel()

# Example texts for prediction
texts_to_predict = [
        "Mit keinem guten Ergebniss",
        "Das war unfair", 
        "Das ist gar nicht mal so gut",
        "Total awesome!",
        "nicht so schlecht wie erwartet"
]

# Make predictions
predictions = model.predict_sentiment(texts_to_predict)
print(predictions)
