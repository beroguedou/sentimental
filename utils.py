import os
import torch
import uuid
import numpy as np
import pandas as pd
from config import *
from datetime import date
from data import text_cleaner, SentimentalInfDataset, SentimentalInfDataLoader
import transformers


def model_loading(model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def load_messages(path):
    df = pd.read_parquet(path)
    sentences_ids = df['ids'].values.tolist()
    sentences = df['messages'].values.tolist()
    return sentences_ids[:15], sentences[:15]


def train_fn(epoch, nb_epochs, model, dataloader, loss_fn, optimizer, print_step=500, device=None):
    model.train()
    running_loss = 0.0
        
    for i, batch in enumerate(dataloader):
        if i == 4000:
            break
        input_texts, labels = batch
        input_texts, labels = input_texts.long().to(device), labels.long().to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = model(input_texts)
        logits = outputs.logits
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if i % print_step == print_step - 1:
            print("Epoch: {}/{}  ---  Iteration: {}  --- Loss: {} ".format(epoch + 1, 
                                                                            nb_epochs,
                                                                            i + 1,
                                                                            round(running_loss/print_step, 4)))
            running_loss = 0.0
      
    
def eval_fn(model, batch_size, dataloader, loss_fn, device=None):
    
    val_loss = 0.0
    val_acc = 0.0
    model.eval()
        
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i == 500:
                break
            input_texts, labels = batch
            input_texts, labels = input_texts.long().to(device), labels.long().to(device)
            # forward 
            outputs = model(input_texts)
            logits = outputs.logits
            # statistics
            # loss
            loss = loss_fn(logits, labels)
            val_loss += loss.item()
            # Accuracy
            probas = torch.softmax(logits, 1)
            _, preds = torch.topk(probas, 1)
            preds = np.hstack(preds.detach().cpu().numpy())
            labels = labels.detach().cpu().numpy()
            good_preds = (np.sum(preds == labels))
            val_acc += good_preds
        val_acc = 100 * val_acc / (batch_size*500)#len(dataloader)               
        current_val_loss = val_loss / 500#len(dataloader)
        print("Validation --> Loss: {}  Accuracy {} %".format(round(current_val_loss, 6), round(val_acc, 2)))
        return current_val_loss
    
    
def batch_inference_fn(model, batch_size, num_workers, max_length, data_path='.', save_path='.', device=None):
    model.eval()
    sentences_ids, sentences = load_messages(data_path) 
    # Create a dataset and a dataloader for inference 
    inf_dataset = SentimentalInfDataset(max_length,
                                        sentences_ids,
                                        sentences)
    
    inf_dataloader = SentimentalInfDataLoader(inf_dataset, 
                                              batch_size,
                                              shuffle=False,
                                              num_workers=num_workers)
    # Prediction loop !
    
    sentences_ids = []
    all_preds = []
    with torch.no_grad():
        for i, batch in enumerate(inf_dataloader):
            input_texts, ids = batch
            input_texts = input_texts.to(device)
            # forward 
            outputs = model(input_texts)
            logits = outputs.logits
            probas = torch.softmax(logits, 1)
            _, preds = torch.topk(probas, 1)
            preds = list(np.hstack(preds.detach().cpu().numpy()))
            all_preds += preds
            sentences_ids += ids
    
    # Formating the predictions and returning back through a parquet file
    predictions = pd.DataFrame()
    predictions['sentences_ids'] = sentences_ids
    #predictions['sentences'] = sentences
    predictions['sentiments'] = all_preds
    # Save the predictions
    today = date.today()
    current_date = today.strftime("%b-%d-%Y")
    save_name = str(uuid.uuid4())+'_'+current_date+'.parquet'
    predictions.to_parquet(os.path.join(save_path, save_name), index=False)
    
    
def one_inference_fn(text, model, tokenizer, device='cuda', max_length=512):
    text = text_cleaner(text)
    encoded_text = torch.tensor(tokenizer.encode(text, max_length=max_length, truncation=True))
    encoded_text = encoded_text.unsqueeze(0).to(device)
    probas = torch.softmax(model(encoded_text).logits, 1)
    value, indice = torch.topk(probas, 1)
    category = indice.detach().cpu().numpy()[0][0]
    verdict = "positive" if category == 1 else "negative"
    value = value.detach().cpu().numpy()[0][0]

    return verdict, value