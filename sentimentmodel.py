from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, AdamW, default_data_collator
from typing import List
import torch
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SentimentModel():
    def __init__(self, model_name: str = "oliverguhr/german-sentiment-bert"):
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'        
            
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.clean_chars = re.compile(r'[^A-Za-züöäÖÜÄß ]', re.MULTILINE)
        self.clean_http_urls = re.compile(r'https*\S+', re.MULTILINE)
        self.clean_at_mentions = re.compile(r'@\S+', re.MULTILINE)

    def fine_tune(self, train_texts: List[str], train_labels: List[str], val_texts: List[str], val_labels: List[str], output_dir: str = "./fine_tuned_model"):
        train_texts, val_texts = [self.clean_text(text) for text in train_texts], [self.clean_text(text) for text in val_texts]
        
        # Encode training and validation data
        train_encodings = self.tokenizer(train_texts, padding=True, truncation=True, return_tensors='pt')
        val_encodings = self.tokenizer(val_texts, padding=True, truncation=True, return_tensors='pt')

        train_dataset = torch.utils.data.TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor([self.model.config.label2id[label] for label in train_labels]))
        val_dataset = torch.utils.data.TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], torch.tensor([self.model.config.label2id[label] for label in val_labels]))

        def custom_data_collator(batch):
            input_ids = torch.stack([item[0] for item in batch])
            attention_mask = torch.stack([item[1] for item in batch])
            labels = torch.stack([item[2] for item in batch])
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            save_total_limit=1,
            save_steps=500,  # Save model every 500 steps
        )

        # Set up Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=custom_data_collator,  # Add this line if not already present
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            optimizers=(AdamW(self.model.parameters(), lr=5e-5), None),
        )

        # Fine-tune the model
        trainer.train()

        # Save the fine-tuned model
        self.model.save_pretrained(output_dir)

    def predict_sentiment(self, texts: List[str], output_probabilities=False) -> List[str]:
            texts = [self.clean_text(text) for text in texts]
            
            # Encode the input texts
            encoded = self.tokenizer.batch_encode_plus(texts, padding=True, truncation=True, return_tensors="pt")
            encoded = encoded.to(self.device)

            with torch.no_grad():
                logits = self.model(**encoded)

            label_ids = torch.argmax(logits[0], axis=1)
            predicted_labels = [self.model.config.id2label[label_id.item()] for label_id in label_ids]

            if output_probabilities:
                predictions = torch.softmax(logits[0], dim=-1).tolist()
                probabilities = [
                    {self.model.config.id2label[index]: item for index, item in enumerate(prediction)}
                    for prediction in predictions
                ]
                result = list(zip(predicted_labels, probabilities))
            else:
                result = predicted_labels

            # Print predictions
            for text, prediction in zip(texts, result):
                print(f"Text: {text}\nPredicted Label: {prediction}\n")

            return result

    def replace_numbers(self,text: str) -> str:
            return text.replace("0"," null").replace("1"," eins").replace("2"," zwei")\
                .replace("3"," drei").replace("4"," vier").replace("5"," fünf") \
                .replace("6"," sechs").replace("7"," sieben").replace("8"," acht") \
                .replace("9"," neun")         

    def clean_text(self,text: str)-> str:    
            text = text.replace("\n", " ")        
            text = self.clean_http_urls.sub('',text)
            text = self.clean_at_mentions.sub('',text)        
            text = self.replace_numbers(text)                
            text = self.clean_chars.sub('', text) # use only text chars                          
            text = ' '.join(text.split()) # substitute multiple whitespace with single whitespace   
            text = text.strip().lower()
            return text