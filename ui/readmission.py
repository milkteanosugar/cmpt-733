# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:54:56 2020

@author: 86152
"""


#参考: https://www.kaggle.com/swarnabha/pytorch-text-classification-torchtext-lstm
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

import torch
from torchtext import data
import torch.nn as nn




def predict(csv1,csv2):

    train = csv1
    test = csv2
    
    #encoding='gb18030'
    
    #print(train.shape)
    
    print('Now loading and predicting........')
    
    
    train_df, valid_df = train_test_split(train)
    

    
    
    import spacy
    spacy_en = spacy.load("en_ner_bionlp13cg_md")
    
    def tokenizer(text): # create a tokenizer function
        
        return [tok.text for tok in spacy_en.tokenizer(text)]
    
    
    TEXT = data.Field(tokenize = tokenizer, include_lengths = True)
    LABEL = data.LabelField(dtype = torch.float)
    
    
    
    class DataFrameDataset(data.Dataset):
    
        def __init__(self, df, fields, is_test=False, **kwargs):
            examples = []
            for i, row in df.iterrows():
                label = row.Label if not is_test else None
                text = row.TEXT
                examples.append(data.Example.fromlist([text, label], fields))
    
            super().__init__(examples, fields, **kwargs)
    
        @staticmethod
        def sort_key(ex):
            return len(ex.text)
    
        @classmethod
        def splits(cls, fields, train_df, val_df=None, test_df=None, **kwargs):
            train_data, val_data, test_data = (None, None, None)
            data_field = fields
    
            if train_df is not None:
                #print('do train')
                train_data = cls(train_df.copy(), data_field, **kwargs)
            if val_df is not None:
                #print('do valid')
                val_data = cls(val_df.copy(), data_field, **kwargs)
            if test_df is not None:
                #print('do test')
                test_data = cls(test_df.copy(), data_field, **kwargs)
    
            return tuple(d for d in (train_data, val_data, test_data) if d is not None)
        
    fields = [('text',TEXT), ('label',LABEL)]
    
    train_ds, val_ds ,test_ds= DataFrameDataset.splits(fields, train_df=train_df, val_df=valid_df,test_df=test)
    
    
    MAX_VOCAB_SIZE = 10000
    
    TEXT.build_vocab(train_ds, 
                     max_size = MAX_VOCAB_SIZE, 
                     vectors = 'glove.6B.50d',
                     unk_init = torch.Tensor.zero_)

    
    LABEL.build_vocab(train_ds)
    
    
    BATCH_SIZE = 64*2
    
    
    device='cpu'
    
    train_iterator, valid_iterator,test_iterator = data.BucketIterator.splits(
        (train_ds, val_ds,test_ds), 
        batch_size = BATCH_SIZE,
        sort_within_batch = True,
        device = device)
    

    
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 50
    HIDDEN_DIM = 50
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = True
    
    DROPOUT = 0.1
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token] # padding
    
    class LSTM_net(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                     bidirectional, dropout, pad_idx):
            
            super().__init__()
            
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
            
            self.rnn = nn.LSTM(embedding_dim, 
                               hidden_dim, 
                               num_layers=n_layers, 
                               bidirectional=bidirectional, 
                               dropout=dropout)
            
            self.fc1 = nn.Linear(hidden_dim*2, hidden_dim)
            
            self.fc2 = nn.Linear(hidden_dim, 1)

            
        def forward(self, text, text_lengths):
            

            embedded = self.embedding(text)

            packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
            
            packed_output, (hidden, cell) = self.rnn(packed_embedded)
            

            
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
            #hidden=hidden[-1,:,:]
            output = self.fc1(hidden)
            output = self.fc2(output)

                
            return output
        
    
    
    
    from sklearn.metrics import roc_auc_score
    
    def binary_accuracy(preds, y):
        """
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
        """

        rounded_preds = (torch.sigmoid(preds)>0.41).float()

    
        
        correct = (rounded_preds == y).float() #convert into float for division 
        acc = correct.sum() / len(correct)
        
        return acc,rounded_preds,torch.sigmoid(preds)
    
    
    
    def evaluate(model, iterator):
        
        epoch_acc = 0
        
        model.eval()
        
        pred_collect=torch.empty(0)
        y_collect=torch.empty(0)
        y_prob=torch.empty(0)
    
        
        
        with torch.no_grad():
            for batch in iterator:
                text, text_lengths = batch.text
                predictions = model(text, text_lengths).squeeze(1)
                

                
                acc,pred_y,prob= binary_accuracy(predictions, batch.label)
                
                epoch_acc = acc.item()+epoch_acc
                pred_collect=torch.cat([pred_collect,pred_y])
                y_collect=torch.cat([y_collect,batch.label])
                y_prob=torch.cat([y_prob,prob])
    
        try:       
            auc=roc_auc_score(y_collect.cpu().data.numpy(),pred_collect.cpu().data.numpy())  
        except:
            auc='UNAVAILABLE'
        return epoch_acc / len(iterator),auc,y_collect,y_prob,pred_collect
    
    
    
    
    model = LSTM_net(INPUT_DIM, 
                EMBEDDING_DIM, 
                HIDDEN_DIM, 
                OUTPUT_DIM, 
                N_LAYERS, 
                BIDIRECTIONAL, 
                DROPOUT, 
                PAD_IDX)
    
    
    
    
    model.load_state_dict(torch.load('LSTM_MODEL',map_location='cpu'))
    
    a,b,my_lab,my_prob,my_pred=evaluate(model, test_iterator)
    return 'Yes, this patient might have readmission' if my_pred.data.numpy() else 'No, this patient might not have readmission' #back to label class

import sys

if __name__ == "__main__":

    train = pd.read_csv(sys.argv[1])
    test = pd.read_csv(sys.argv[2],encoding='gb18030')
    result=predict(train,test)
    print('our prediction of that single input is...', result)
    


