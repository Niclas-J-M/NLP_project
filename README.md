# NLP_project

This repository contains the code and data for fine-tuning the [German Sentiment Classification with BERT](https://github.com/oliverguhr/german-sentiment) model. 

## Data Sets
The dataset has in total 1339 samples which are randomly selected in data_generation.py. The amount of sample that you want to use can be adjusted in train_sentiment_model.py in the line:
```python
texts, labels = generate_texts_labels(num_samples=500). 
``` 


## Install 

To run the code install the package from [pypi](https://pypi.org/project/germansentiment/):

```bash
pip install germansentiment
pip install transformers
```

## Usage

To run the code and produce the F1 scores and results call the train_sentiment_model.py in the terminal using python. (Our python version that we used is 3.11.3)

Calling train_sentiment_model will call data_generation.py returning a random selection of data samples where the amount of samples can be set in data_generation.py. This will return the texts and labels which are further split into a train-test split with 0.8 and 0.2 (This can be changed in train_sentimment_model.py). This will further call the non-fine-tuned and the fine-tuned model in sentimentmode.py. At the end the results are returned and an F1 score is calculated and reported in train_sentiment_model.py and shown in the terminal.