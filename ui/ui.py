import pathlib
import os

import pandas as pd
import numpy as np

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import re
import nltk
import dash_dangerously_set_inner_html as inner_html
import sys
import predict
import pickle


los_col = ['ICUSTAY', 'DRUG', 'Num_CPT', 'REQUEST_TELE', 'REQUEST_RESP',
       'REQUEST_CDIFF', 'REQUEST_MRSA', 'REQUEST_VRE', 'TRANSFERTIME',
       'Blood Diseases', 'Circulatory System', 'Congenital Anomalies',
       'Digestive System', 'Endocrine, Immunity Disorders',
       'Genitourinary System', 'Infectious Diseases', 'Injury',
       'Mental Disorders', 'Musculoskeletal System', 'Neoplasms',
       'Nervous System', 'Perinatal Period', 'Pregnancy', 'Respiratory System',
       'Skin and Subcutaneous Tissue', 'Symptoms Conditions', "STOP_D/C'd",
       'STOP_NotStopd', 'STOP_Restart', 'STOP_Stopped', 'EVEN_admit',
       'EVEN_discharge', 'EVEN_transfer', 'SER_CMED', 'SER_CSURG', 'SER_MED',
       'SER_NMED', 'SER_NSURG', 'SER_OMED', 'SER_OTHER', 'SER_SURG',
       'SER_TRAUM', 'SER_VSURG', 'CARE_ICU', 'AGE_newborn', 'AGE_teens',
       'AGE_young-adult', 'AGE_adult', 'AGE_senior', 'DB_carevue',
       'DB_carevue & metavision', 'DB_metavision', 'GENDER_F', 'GENDER_M',
       'INSUR_Government', 'INSUR_Medicaid', 'INSUR_Medicare', 'INSUR_Private',
       'INSUR_Self Pay', 'RELI_CATHOLIC', 'RELI_JEWISH', 'RELI_NOT SPECIFIED',
       'RELI_OTHER', 'RELI_PROTESTANT QUAKER', 'RELI_UNOBTAINABLE',
       'MARR_DIVORCED', 'MARR_LIFE PARTNER', 'MARR_MARRIED', 'MARR_OTHER',
       'MARR_SEPARATED', 'MARR_SINGLE', 'MARR_UNKNOWN (DEFAULT)',
       'MARR_WIDOWED', 'ETH_ASIAN', 'ETH_BLACK/AFRICAN AMERICAN',
       'ETH_HISPANIC OR LATINO', 'ETH_OTHER', 'ETH_UNKNOWN/NOT SPECIFIED',
       'ETH_WHITE']

mort_col = ['age', 'NumCallouts', 'NumDiagnosis', 'NumProcs', 'NumCPTevents', 'NumInput', 'NumOutput', 'NumLabs', 'NumMicroLabs',
            'NumNotes', 'NumProcEvents', 'NumTransfers', 'NumChartEvents', 'NumRx', 'TotalNumInteract', 'F', 'M', 'ELECTIVE', 'EMERGENCY',
            'NEWBORN', 'URGENT', '** INFO NOT AVAILABLE **', 'CLINIC REFERRAL/PREMATURE', 'EMERGENCY ROOM ADMIT', 'HMO REFERRAL/SICK',
            'PHYS REFERRALRMAL DELI', 'TRANSFER FROM HOSP/EXTRAM', 'TRANSFER FROM OTHER HEALT', 'TRANSFER FROM SKILLED NUR', 'TRSF WITHIN THIS FACILITY']

all_patient_df = pd.read_csv('data/patient_info.csv')

patient_col = ['Med_ID']+mort_col+los_col
patient_data = [0] * len(patient_col)
patient_df = pd.DataFrame([patient_data], columns=patient_col)

# app initialize
app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
server = app.server
app.config["suppress_callback_exceptions"] = True

# Load data
APP_PATH = str(pathlib.Path(__file__).parent.resolve())

# the app lay out, style, etc
app.layout = html.Div(
    children=[
        html.Div(
            id="banner",
            className="banner",
            children=[dbc.NavbarBrand("Natural Language Processing", className="ml-2")],
        ),
        html.Div(
            id="top-row",
            className="row",
            children=[
                html.Div(
                    className="row",
                    id = 'top-row-info',
                    style={
                        'margin-top': '50px',
                        'padding': '10px',
                        'background-color': 'lightblue',
                        'width':'30%',
                        'max-height': 'calc(100vh - 10.5rem)',
                    },
                    children = [
                        html.H4('Search Patient'),
                        dcc.Input(
                            id="input1",
                            type="number",
                            placeholder="Medical ID",
                            style = {
                                'margin-top':"20px",
                                'width':'100%',
                                'display': 'inline',
                            }
                        ),
                        html.Div(
                            children = [
                                dbc.Button("Retrieve",
                                    id = "Retrieve",
                                    color="primary",
                                    style = {
                                        'margin-top':"10px",
                                        'float':'left',
                                        # 'margin-right':'10%',
                                        'background-color':'white',
                                        'width':'40%'
                                    }
                                ),
                                dbc.Button("CreateNew",
                                   id="CreateNew",
                                   color="primary",
                                   style={
                                       'margin-top': "10px",
                                       'float': 'right',
                                       # 'margin-right': '10%',
                                       'background-color': 'white',
                                        'width':'50%'
                                   }
                                ),
                            ]
                        ),
                        html.Div(
                            style = {
                                'margin-top':'50px',
                                'padding':'10px'
                            },
                            children=[
                                html.H4("Information:"),
                                html.P(id='retoutput', children = []),
                                html.P(id='createnew', children = [])
                            ],
                        ),
                    ]
                ),
                html.Div(
                    className="row",
                    id="top-row-header",
                    children=[
                        html.Div(
                            id="header-container",
                            children=[
                                html.H4(
                                    'Please enther your note:',
                                    style = {
                                        'color':'black',
                                        'padding-left':'20px'
                                    },
                                ),
                                dcc.Textarea(
                                    id="textinput",
                                    className="four columns",
                                    placeholder="Start your notes...",
                                    value = '',
                                    style = {
                                        'margin-top':' 20px',
                                        'padding': '0rem 2rem',
                                        'min-height': 'calc(100vh - 30.5rem)',
                                        'min-width': 'calc(100vh - 20.5rem)',
                                        'overflow-y': 'auto',
                                        'border-radius': '4px'
                                    },
                                    # children=[html.Div(dcc.Input(id='input-box', type='text'))]
                                ),
                                html.Div(
                                    children = [
                                        dcc.Upload(
                                            id="upload-data",
                                            children=html.Div([
                                                'Drag and Drop or ',
                                                html.A('Select Files')
                                            ]),
                                            style={
                                                'display': 'inline',
                                                'float': 'left',
                                                'width': '25rem',
                                                'lineHeight': '25px',
                                                'borderStyle': 'dashed',
                                                'borderRadius': '5px',
                                                'textAlign': 'center',
                                                'margin': '25px',
                                                'border-radius':'4px',

                                            },
                                            # Allow multiple files to be uploaded
                                            multiple=False
                                        ),
                                        html.Button(
                                            'Submit',
                                            id="getword",
                                            style={
                                                'margin-top': ' 20px',
                                                'margin-right': '10px',
                                                'padding': '0rem 2rem',
                                                'color': 'black',
                                                'display': 'inline',
                                                'float': 'right',
                                                'width': '30%'
                                            },
                                        ),
                                    ]
                                ),
                            ],
                        )
                    ],
                ),
                html.Div(
                    className="top-row",
                    # id="top-row-graphs",
                    style={
                        'width': '800px',
                        'margin-top': '50px',
                        'word-wrap':'break-word',
                    },
                    children=[
                        dbc.Jumbotron(
                            [html.H2("Medical Volcabulary", className="display-3"),
                             html.Div(
                                 style={'display': 'flex','flex-direction':'row', 'padding':'2px'},
                                 children = [
                                    dcc.Checklist(
                                        id = 'bchem',
                                        options=[
                                            {'label': 'B-Chemistry', 'value': 'B-Chemistry'},
                                        ],
                                        value=['B-Chemistry'],
                                        labelStyle={'margin':'2px','background-color':'#00FEFE'},
                                    ),
                                    dcc.Checklist(
                                        id = 'ichem',
                                        options=[
                                            {'label': 'I-Chemistry', 'value': 'I-Chemistry'},
                                        ],
                                        value=['I-Chemistry'],
                                        labelStyle={'margin':'2px','background-color':'#FF6347'},
                                    ),
                                    dcc.Checklist(
                                        id = 'idise',
                                        options=[
                                            {'label': 'I-Disease', 'value': 'I-Disease'},
                                        ],
                                        value=['I-Disease'],
                                        labelStyle={'margin':'2px','background-color':'#FFD700'},
                                    ),
                                    dcc.Checklist(
                                        id = 'bdise',
                                        options=[
                                            {'label': 'B-Disease', 'value': 'B-Disease'},
                                        ],
                                        value=['B-Disease'],
                                        labelStyle={'margin':'2px','background-color':'#8FBC8F'},
                                    ),

                                 ]
                             ),


                            html.Div(
                                children = [
                                    html.P(
                                        id = 'prediction',
                                        style = {
                                            # 'verflow-x': 'auto',
                                            'overflow-y': 'auto',
                                            'overflow-x': 'hidden',
                                            'height': 'calc(100vh - 47rem)',
                                        },
                                        children = [],
                                    ),
                                    html.H4("Implement for notes"),
                                    dcc.Checklist(
                                        id = 'impliements',
                                        labelStyle = {
                                           'display': "inline-block",
                                            'margin': '2px',
                                        },
                                        options = [{'label':'URGENT','value':'URGENT'},
                                                    {'label':'CLINIC REFERRAL/PREMATURE','value':'CLINIC REFERRAL/PREMATURE'},
                                                    {'label':'EMERGENCY ROOM ADMIT','value':'EMERGENCY ROOM ADMIT'},
                                                    {'label':'HMO REFERRAL/SICK','value':'HMO REFERRAL/SICK'},
                                                    {'label':'PHYS REFERRALRMAL DELI','value':'PHYS REFERRALRMAL DELI'},
                                                    {'label':'TRANSFER FROM HOSP/EXTRAM','value':'TRANSFER FROM HOSP/EXTRAM'},
                                                    {'label':'TRANSFER FROM OTHER HEALT','value':'TRANSFER FROM OTHER HEALT'},
                                                    {'label':'TRANSFER FROM SKILLED NUR','value':'TRANSFER FROM SKILLED NUR'},
                                                    {'label':'TRSF WITHIN THIS FACILITY','value':'TRSF WITHIN THIS FACILITY'},
                                        ],
                                        value = []
                                    ),
                                    html.Div(id='implementhide',children = [])
                                ]

                            ),
                            html.P(dbc.Button("Prediction", id = '3predict', color="primary"), className="lead"),
                             html.P(id = 'save_patient',children = ''),]

                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            className="row",
            # id="bottom-row",
            style = {
                'display': 'flex',
                'flex-direction': 'row',
                'flex-wrap': 'nowrap',
                'width': '100%'
            },
            children=[
                html.Div(
                    id = 'LOS',
                    style = {
                        'width': '100%'
                    },
                    children=[],
                ),
                html.Div(
                    id = 'mortality',
                    style={
                        'width': '100%'
                    },
                    children=[
                    ],
                ),
                html.Div(
                    id = 'readmission',
                    style={
                        'width': '100%'
                    },
                    children=[
                    ],
                ),
            ],
        ),
    ]
)


@app.callback(
    Output('save_patient','children'),
    [Input('3predict','n_clicks')]
)
def update_allpatient(click):
    ret = ''
    if click:
        print(patient_df)
        all_df = pd.concat([all_patient_df, patient_df]).drop_duplicates(['Med_ID'],keep='last').reset_index(drop=True)
        all_df.to_csv(r'data/patient_info.csv',index = False)
    return ret


############################ update implement ####################################
# the implement check box under the name entities
@app.callback(
    Output('implementhide','children'),
    [Input('3predict','n_clicks'),
     Input('impliements','value'),]
)
def get_implement(click, values):
    global patient_df
    ret = ''
    if click:
        for value in values:
            patient_df[value] = 1


    return ret

########################### update disease ########################################
# update the current patient_df, by comparing the current extract medical words
# if the words assist the prediction, update the patient profile
def update_prediction(med_words):
    global patient_df
    Idisease = med_words['I-Disease']
    Bdisease = med_words['B-Disease']
    for i in los_col:
        d_word = nltk.word_tokenize(i)
        for j in Idisease:
            if j in d_word:
                patient_df[i] = 1

        for j in Bdisease:
            if j in d_word:
                patient_df[i] = 1

    return patient_df

########################### collect info ####################################
# for new patient, create the profile
@app.callback(
    Output('createnew','children'),
    [Input('CreateNew','n_clicks'),
     Input('Retrieve','n_clicks'),]
)
def get_createForm(click,retr):
    form = ''
    if retr:
        form = ''
    if click:
        form = html.Div(
            id = 'newpatient',
            children = [
                dcc.Input(id = 'name', style = {'margin':'5px', 'width':'100%','display':'inline-block'}, placeholder = 'Name'),
                dcc.Input(id='age', type = "number", style = {'margin':'5px', 'width':'100%','display':'inline-block'},placeholder='Age'),
                dcc.Dropdown(id = 'gender',
                             options = [{'label':'female', 'value':0,},{'label':'male', 'value':1}],
                             style = {'margin': '5px 0 0px 2px', 'width':'100%','display':'inline-block'},
                             placeholder = 'gender' ),
                dcc.Dropdown(id = 'marriage',
                             style={'margin': '5px 0 0px 2px', 'width': '100%','display':'inline-block'},
                             options = [
                                 {'label':'Divorced', 'value':0,},
                                 {'label':'Lift partner', 'value':1},
                                 {'label': 'Married', 'value': 2},
                                 {'label': 'Single','value':3},
                                 {'label': 'Separated','value':4},
                                 {'label': 'Other','value':5},
                                 # {'label': 'Unknown', 'value': 6},
                             ],
                             value = 6,
                             placeholder = 'marrige'),
                dcc.Dropdown(id = 'ethic',
                             style={'margin': '5px 0 0px 2px', 'width': '100%','display':'inline-block'},
                             options=[
                                 {'label': 'Asian', 'value': 0, },
                                 {'label': 'Black american', 'value': 1},
                                 {'label': 'Latino', 'value': 2},
                                 {'label': 'Other', 'value': 3},
                                 {'label': 'white', 'value': 5},
                             ],

                             value = 4,
                             placeholder = 'Ethic'),
                dcc.Dropdown(id='religion',
                             style={'margin': '5px 0 5px 2px', 'width': '100%','display':'inline-block'},
                             options=[
                                 {'label': 'Catholic', 'value': 0, },
                                 {'label': 'Jewish', 'value': 1},
                                 {'label': 'Not specified', 'value': 2},
                                 {'label': 'Other', 'value': 3},
                                 {'label': 'Protestant quaker', 'value': 4},
                                 {'label': 'Unobtainable', 'value': 5},
                             ],
                             value=6,
                             placeholder='Religion'),
                html.Div(id = 'hidden',children = [])
            ]
        ),

    return form

# read the new patient's profile from input
@app.callback(
    Output('hidden','children'),
    [Input('gender','value'),
    Input('name','value'),
    Input('age','value'),
    Input('marriage','value'),
    Input('ethic','value'),
    Input('religion','value'),
     ]
)
def get_info(gender, name,age,marry,ethic, religion):
    global patient_df
    ret = ''
    if gender is 1:
        patient_df['M'] = 1
        patient_df['GENDER_M'] = 1
    else:
        patient_df['F'] = 1
        patient_df['GENDER_F'] = 1

    if name:
        patient_df['Med_ID'] = len(patient_df['Med_ID'])+1
        # patient_df.set_index('Med_ID')

    if age:
        patient_df['age'] = age
        # los table for age
        if int(age) <= 1:
            patient_df['AGE_newborn'] = 1
        elif int(age) <= 18 and int(age) > 1:
            patient_df['AGE_teens'] = 1
        elif int(age) <= 30 and int(age) > 18:
            patient_df['AGE_young-adult'] = 1
        elif int(age) <= 60 and int(age) > 30:
            patient_df['AGE_adult'] = 1
        else:
            patient_df['AGE_senior'] = 1

    if marry:
        if int(marry) == 0:
            patient_df['MARR_DIVORCED'] = 1
        elif int(marry) == 1:
            patient_df['MARR_LIFE PARTNER'] = 1
        elif int(marry) == 2:
            patient_df['MARR_MARRIED'] = 1
        elif int(marry) == 3:
            patient_df['MARR_MARRIED'] = 1
        elif int(marry) == 4:
            patient_df['MARR_SINGLE'] = 1
        elif int(marry) == 5:
            patient_df['MARR_SEPARATED'] = 1
        elif int(marry) == 6:
            patient_df['MARR_UNKNOWN (DEFAULT)'] = 1

    if ethic:
        if int(ethic) == 0:
            patient_df['ETH_ASIAN'] = 1
        elif int(ethic) == 1:
            patient_df['ETH_BLACK/AFRICAN AMERICAN'] = 1
        elif int(ethic) == 2:
            patient_df['ETH_HISPANIC OR LATINO'] = 1
        elif int(ethic) == 3:
            patient_df['ETH_OTHER'] = 1
        elif int(ethic) == 4:
            patient_df['ETH_UNKNOWN/NOT SPECIFIED'] = 1
        elif int(ethic) == 5:
            patient_df['ETH_WHITE'] = 1

    if religion:
        if int(religion) == 0:
            patient_df['RELI_CATHOLIC'] = 1
        elif int(religion) == 1:
            patient_df['RELI_JEWISH'] = 1
        elif int(religion) == 2:
            patient_df['RELI_NOT SPECIFIED'] = 1
        elif int(religion) == 3:
            patient_df['RELI_OTHER'] = 1
        elif int(religion) == 4:
            patient_df['RELI_PROTESTANT QUAKER'] = 1
        else:
            patient_df['RELI_UNOBTAINABLE'] = 1
    return ret


########################## retrieve patient info #############################
# if the patient already exist, retrieve it's information from patient_info.csv
@app.callback(
    Output('retoutput','children'),
    [
        Input('input1','value'),
        Input('Retrieve','n_clicks'),
        Input('CreateNew','n_clicks')
    ]
)
def get_information(inputs, click, creatnew):
    string = ''
    if click and inputs:
        if int(inputs) in all_patient_df['Med_ID'].tolist():
            global patient_df
            patient_df = all_patient_df.loc[all_patient_df['Med_ID'] == int(inputs)]
            if int(patient_df['GENDER_F']) is 1:
                gender = 'female'
            else:
                gender = 'male'
            string = '''
            <p><b>Name: </b>'''+ str(patient_df['Med_ID'].values[0])+'''</p>
            <p><b>Age</b>: '''+ str(patient_df['age'].values[0])+'''</p>
            <p><b>Gender: </b>'''+ gender +'''</p>
            '''
        else:
            string = '''<p><b>Patient Not Exist, please create new </b>'''
    if creatnew:
        string = ''
    return html.Div(
        inner_html.DangerouslySetInnerHTML(string)
    )

################ predict the mortality##################
@app.callback(
    Output('mortality','children'),
    [Input('3predict','n_clicks')]
)
def get_motality(click):
    ret = "Mortality: "
    mort_df = patient_df[mort_col]
    if click:
        model = pickle.load(open('pickle_model.pkl', 'rb'))
        pred = model.predict(mort_df)
        if pred is 1:
            ret = ret + 'high likely to dead'
        else:
            ret = ret + 'high likely to be cured'
    return [dbc.Col(html.H3(ret), style={'padding': '50px'}),
            dbc.Col(html.Img(src="assets/MORTALITY.jpg", height="300 px"), style={'padding-left': '50px'}),]

################ predict the length of stay ##################
@app.callback(
    Output('LOS','children'),
    [Input('3predict','n_clicks')]
)
def get_LOS(click):

    ret = "Length of Stay: "
    los_df = patient_df[los_col]
    if click:
        rmodel = pickle.load(open('reg_model', 'rb'))
        rpred = rmodel.predict(los_df)
        ret = 'Patient possible staty ' + str(rpred[0]) + ' days'
    return [dbc.Col(html.H3(ret), style={'padding': '50px'}),
            dbc.Col(html.Img(src="assets/STAY.jpg", height="300 px"), style={'padding-left': '50px'}),]

################ read upload #######################
def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'txt' in filename:
            f = open(io.StringIO(decoded.decode('utf-8')))
            data = f.read()

    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.P(data)
    ])


@app.callback(Output('textinput', 'value'),
              [Input('upload-data', 'contents')])
def update_output(content):
    if content is not None:
        return content


####################### reads the inputs from textarea ##################
def get_medwords(words):
    B_Chemistry= []
    I_Chemistry = []
    B_Disease = []
    I_Disease = []
    for i in words:
        word,tag = nltk.word_tokenize(i)
        if tag == 'B-Chemistry':
            B_Chemistry.append(word)
        elif tag == 'I-Chemistry':
            I_Chemistry.append(word)
        elif tag == 'B-Disease':
            B_Disease.append(word)
        elif tag == 'I-Disease':
            I_Disease.append(word)
    med_words = {'B-Chemistry':B_Chemistry, 'I-Chemistry':I_Chemistry, 'B-Disease': B_Disease, 'I-Disease': I_Disease}
    update_prediction(med_words)
    return med_words


# get the name entities from transfermation-ner
med_words = []
@app.callback(
    Output('prediction','children'),
    [Input('textinput','value'),
     Input('getword','n_clicks'),
     Input('bchem','value'),
    Input('ichem','value'),
    Input('bdise','value'),
    Input('idise','value'),
     ]
)
def prediction(inputs,click,bchem,ichem,bdise,idise):
    string = ''
    if inputs and click:
        # token pass to ner transform function replace read
        global med_words
        med_words = predict.predict(inputs)
        med_words = get_medwords(med_words)

        paragraphs = inputs.split('/n')
        for paragraph in paragraphs:
            string = string + '''<br>'''
            # change color of the word if the word in med words
            words = nltk.word_tokenize(paragraph)
            for word in words:
                if word in med_words['B-Chemistry'] and 'B-Chemistry' in bchem:
                    string = string + ''' <span style="background-color:#00FEFE">''' + word+ '''</span>'''
                elif word in med_words['I-Chemistry'] and 'I-Chemistry' in ichem:
                    string = string + ''' <span style="background-color:#FF6347">''' + word + '''</span>'''
                elif word in med_words['B-Disease'] and 'B-Disease' in bdise:
                    string = string + ''' <span style="background-color:#FFD700">''' + word + '''</span>'''
                elif word in med_words['I-Disease'] and 'I-Disease' in idise:
                    string = string + ''' <span style="background-color:#8FBC8F">''' + word + '''</span>'''
                else:
                    string = string + ' ' + word
    return html.Div(
        inner_html.DangerouslySetInnerHTML(string)
    )


# Running the server
if __name__ == "__main__":
    app.run_server(debug=True)
