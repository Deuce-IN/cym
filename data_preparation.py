import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizerFast

def load_and_preprocess_data(file_path):
   
    data = pd.read_csv("/Users/rahul/Desktop/Project CYMBA/Conversation.csv")
    
    
    data['text'] = data['question'] + " [SEP] " + data['answer']  
    data['label'] = 1  
    
   
    data = data[['text', 'label']]

    
    dataset = Dataset.from_pandas(data)

    
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

   
    def tokenize_function(example):
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

  
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    return tokenized_dataset

if __name__ == "__main__":
    
    csv_file_path = "/Users/rahul/Desktop/Project CYMBA/Conversation.csv"
    
   
    tokenized_data = load_and_preprocess_data(csv_file_path)
    
   
    save_path = "./tokenized_data"  
    

    tokenized_data.save_to_disk(save_path)
     
    print(f"Tokenized data saved at {save_path}")
