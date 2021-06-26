import transformers
from datasets import load_dataset



# Some variable that we will use everywhere in the code
max_length = 128
tokenizer = transformers.CamembertTokenizer.from_pretrained("camembert-base")
# Taking the dataset, the model and its tokenizer from HuggingFace 
dataset = load_dataset("allocine")
#model = transformers.CamembertForSequenceClassification.from_pretrained("camembert-base")