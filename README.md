# ClINICAL BIG DATA RESEARCH
In this project, we studied four questions in clinical researches, as described by following.
## Medical Name Entity Recognition
Given a medical note, we want extract medical and disease entities. In this question, we make a model that takes a note string as input and tag each word as one of disease or chemical.

Detailed Instructions on how to train or download trained model can be read [here](https://csil-git1.cs.surrey.sfu.ca/britneyt/cmpt733-project/-/blob/master/transfermation-ner/README.md)

To run prediction, excute following script:
```bash
from predict import predict
note = 'This is a test note'
results = predict(note)
```
Results will be returned as list od words followed by a tag.


## Mortality Prediction
We covered the clinical problems of predicting Mortality Rate

We uploaded the mimic-3 database to google drive and run pyspark on google colab to do preprocessing.

The preprocessing result table was saved as a .csv file. This mimic3_prev.csv was pushed under the mortality rate prediciont dirct on gitlab.

The modelling and feature engineering part have been done with pandas and numpy.

For detailed instructions, please refer to the comments in [this jupyter notebook](https://csil-git1.cs.surrey.sfu.ca/britneyt/cmpt733-project/-/blob/master/mortality_prediction/Mortality_prediction.ipynb) for mortality prediction

## Length of Stay Prediction
We covered the clinical problems of forecasting length of stay (LOS).  

Each person’s LOS is defined based on the number of days between their admission and discharge from the hospital.

We used both Regression (exact number of days) and Classification (time frame) to predict LOS.

Please refer to [this jupyter notebook](https://csil-git1.cs.surrey.sfu.ca/britneyt/cmpt733-project/-/blob/master/length_of_stay/Predict_LOS.ipynb) for more details.

To run prediction:
```bash
python length_of_stay/predict_los.py
```
An example of inputs:
394, 205, 63, TELE, 1, Endocrine, Immunity Disorders:2&Injury:4&Nervous System:2&Respiratory System:3&Skin and Subcutaneous Tissue:3, 
Restart, discharge, TRAUM, 1, young-adult, carevue & metavision, M, Medicaid, NOT SPECIFIED, SINGLE, WHITE

Both exact number of day and time range will be printed.

## Readmission Prediction


First, Thanks to Nwamaka's medium post(https://medium.com/nwamaka-imasogie/clinicalbert-using-deep-learning-transformer-model-to-predict-hospital-readmission-c82ff0e4bb03), 
which offers the data preprocessing stage as well as some guidance on classification. The following project will read the data from the pre-processed stage, 
and perform the machine learning techniques on Readmission prediction.
 
In this [notebook](https://csil-git1.cs.surrey.sfu.ca/britneyt/cmpt733-project/-/blob/master/NLP_on_readmission/NLP_readmission.ipynb), our team performed our own research on NLP processing. This was a supplementary research on pure text processing.
The detailed step-by-step instructions were shown in "NLP_readmission.ipynb"


## UI
We design a web applcation to display our results.

### Installing
In order to run the UI, please follow the indicated steps

Install libraries
```
pip install -r requirements.txt

#For transformers specific
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
```
Download trained NER model from 
```
https://drive.google.com/file/d/1eidhQ0i9NHkXqFKoS2GhdZDlJceJQvDj/view
```

Detailed Instructions on how to train or download trained model can be read [here](https://csil-git1.cs.surrey.sfu.ca/britneyt/cmpt733-project/-/blob/master/transfermation-ner/README.md)
Please download the model and extract it as transformation-ner/output
### running code

```
python ui.py

```

### Using instruction:
* retrieve or create a new patient profile by clicking button at top left box (suggest trying id: 1)
* write or upload new file in the he top middle text area, then click submit; suggest note:
```
ADDENDUM:

RADIOLOGIC STUDIES:  Radiologic studies also included a chest
CT, which confirmed cavitary lesions in the left lung apex
consistent with infectious process/tuberculosis.  This also
moderate-sized left pleural effusion.

HEAD CT:  Head CT showed no intracranial hemorrhage or mass
effect, but old infarction consistent with past medical
history.

ABDOMINAL CT:  Abdominal CT showed lesions of
T10 and sacrum most likely secondary to osteoporosis. These can
be followed by repeat imaging as an outpatient.

```
* click any implementation message
* click predict