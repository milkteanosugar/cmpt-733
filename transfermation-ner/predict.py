import sys
import re
import os

def predict_helper(note_tokens,start,end):
  #Save to output/test.txt
  f = open('data-input/test.txt','w')
  f.writelines(note_tokens[start:end])
  f.close()
  os.system('python ./run_ner.py --data_dir ./data-input --model_type bert --model_name_or_path output/ --output_dir ./output --labels ./data/labels.txt --do_predict --max_seq_length 256 --overwrite_output_dir --overwrite_cache')
  #read the file
  f = open('output/test_predictions.txt','r')
  sub_results = f.readlines()
  f.close()
  return sub_results


def predict(note):
    #Split the word into tokens by punctuation
    note_tokens = re.split(r'\W+',note)
    #Add a dummy tag 'O' after each word
    note_tokens = [token + ' ' + 'O\n'for token in note_tokens] 
    n =len(note_tokens)

    #Since the model can only handle 256 words maximum
    #workaround to handle longer notes
    c,r = divmod(n,256)
    i = 0
    results = []
    while i < c:
        print('now i = {}'.format(i))
        results+=predict_helper(note_tokens,i*256,(i+1)*256)
        i+=1
    #Handle the rest
    results += predict_helper(note_tokens,i*256,n)
    return results

#Download the model,comment out if you already have it 
#os.system('wget https://www.dropbox.com/s/x5ln8z1e0dp2iwd/discharge_ner_model.zip?dl=0')

