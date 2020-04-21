


import sys
import re
import os

def predict_helper(note_tokens,start,end):
  #Save to output/test.txt
  f = open('../transfermation-ner/data-input/test.txt','w')
  f.writelines(note_tokens[start:end])
  f.close()
  os.system('python ../transfermation-ner/run_ner.py --data_dir ../transfermation-ner/data-input --model_type bert --model_name_or_path ../transfermation-ner/checkpoint-1700_disch/ --output_dir ../transfermation-ner/output --labels ../transfermation-ner/data/labels.txt --do_predict --max_seq_length 256 --overwrite_output_dir --overwrite_cache')
  #read the file
  f = open('../transfermation-ner/output/test_predictions.txt','r')
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

# note = '''
# Service:
# ADDENDUM:
#
# RADIOLOGIC STUDIES:  Radiologic studies also included a chest
# CT, which confirmed cavitary lesions in the left lung apex
# consistent with infectious process/tuberculosis.  This also
# moderate-sized left pleural effusion.
#
# HEAD CT:  Head CT showed no intracranial hemorrhage or mass
# effect, but old infarction consistent with past medical
# history.
#
# ABDOMINAL CT:  Abdominal CT showed lesions of
# T10 and sacrum most likely secondary to osteoporosis. These can
# be followed by repeat imaging as an outpatient.
#
#
#
#                             [**First Name8 (NamePattern2) **] [**First Name4 (NamePattern1) 1775**] [**Last Name (NamePattern1) **], M.D.  [**MD Number(1) 1776**]
#
# '''
# predict(note)