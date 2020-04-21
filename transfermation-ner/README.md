# Medical Name Entity Recognition

## Installation

### Prerequisites

* Python ≥ 3.6

### Provision a Virtual Environment

If `pip` is configured in your conda environment, 
install dependencies from within the project root directory
```
pip install -r requirements.txt
``` 

### Data Pre-processing

#### Download the data
The current `BC5CDR` dataset is available as IOB format. Small modifications should be applied 
to the files so they can be processed by BERT NER (space separated elements, etc.). 
We will first download the files and then transform them

Download the files at:
```bash
mkdir data-input
curl -o data-input/devel.tsv https://raw.githubusercontent.com/cambridgeltl/MTL-Bioinformatics-2016/master/data/BC5CDR-IOB/devel.tsv
curl -o data-input/train.tsv https://raw.githubusercontent.com/cambridgeltl/MTL-Bioinformatics-2016/master/data/BC5CDR-IOB/train.tsv
curl -o data-input/test.tsv https://raw.githubusercontent.com/cambridgeltl/MTL-Bioinformatics-2016/master/data/BC5CDR-IOB/test.tsv

```

To transform the data in a BERT NER compatible format, execute the following command:
```bash
python ./preprocess/generate_dataset.py --input_train_data data-input/train.tsv --input_dev_data data-input/devel.tsv --input_test_data data-input/test.tsv --output_dir data-input/
```

The script ouputs two files `train.txt` and `test.txt` that will be the input of the NER pipeline.

### Download ClinicalBert pre-trained model and run the NER task
Download ClinicalBert model via this [github link](https://github.com/EmilyAlsentzer/clinicalBERT). We use biobert_pretrain_output_all_notes_150000 for fine-tuning
To execute the NER pipeline, run the following scripts:
```bash
python ./run_ner.py --data_dir ./data-input --model_type bert --model_name_or_path bert_pretrain_output_all_notes_150000/ --output_dir ./output --labels ./data/labels.txt --do_train --do_predict --max_seq_length 256 --overwrite_output_dir --overwrite_cache
```
The script will output the results and predictions in the output directory.

### Download trained NER model for prediction
For this project, we provide [model link](https://drive.google.com/open?id=1eidhQ0i9NHkXqFKoS2GhdZDlJceJQvDj) to download. Please download it and extract it as transformation-ner/output.

An example script to run the prediction
```bash
from predict import predict
note = 'This is a test note'
results = predict(note)
```
Results will be returned as list od words followed by a tag.