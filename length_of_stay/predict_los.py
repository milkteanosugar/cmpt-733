import pandas as pd
import numpy as np
import pickle


lst1 = pd.DataFrame([], columns = ['ICUSTAY_ID'])
ICUSTAY_ID = input('Enter ICUSTAY days: ')
lst1 = lst1.append({'ICUSTAY_ID': int(ICUSTAY_ID)}, ignore_index=True)

lst2 = pd.DataFrame([], columns = ['DRUG'])
d = input('Enter DRUG number: ')
lst2 = lst2.append({'DRUG': int(d)}, ignore_index=True)

lst4 = pd.DataFrame([], columns = ['Num_CPT'])
c = input('Enter CPT number: ')
lst4 = lst4.append({'Num_CPT': int(c)}, ignore_index=True)

a = np.zeros(shape=(1,5))
lst5 = pd.DataFrame([], columns = ["REQUEST_TELE", 'REQUEST_RESP', 'REQUEST_CDIFF', 'REQUEST_MRSA', 'REQUEST_VRE'])
g = input("Enter STOPPED type ('TELE', 'RESP', 'CDIFF', 'MRSA', 'VRE', 'None'): ")
if g != 'None':
    lst5 = lst5.append({'REQUEST_{}'.format(str(g)): int(1)}, ignore_index=True)
else:
    lst5 = pd.DataFrame(a, columns = ["REQUEST_TELE", 'REQUEST_RESP', 'REQUEST_CDIFF', 'REQUEST_MRSA', 'REQUEST_VRE'])

lst6 = pd.DataFrame([], columns = ['TRANSFERTIME'])
g = input('Enter TRANSFERTIME: ')
lst6 = lst6.append({'TRANSFERTIME': int(g)}, ignore_index=True)   


a = np.zeros(shape=(1,17))
col = ['Blood Diseases', 'Circulatory System', 'Congenital Anomalies',
       'Digestive System', 'Endocrine, Immunity Disorders',
       'Genitourinary System', 'Infectious Diseases', 'Injury',
       'Mental Disorders', 'Musculoskeletal System', 'Neoplasms',
       'Nervous System', 'Perinatal Period', 'Pregnancy', 'Respiratory System',
       'Skin and Subcutaneous Tissue', 'Symptoms Conditions']
lst7 = pd.DataFrame([], columns = col)
g = input("""Enter one of the ICD9 diagnosis, if more than one, split by '&', no space between, follow the number in ':' 
        'Blood Diseases', 'Circulatory System', 
        'Congenital Anomalies','Digestive System', 
        'Endocrine, Immunity Disorders',
       'Genitourinary System', 'Infectious Diseases',
       'Injury', 'Mental Disorders', 'Musculoskeletal System', 
       'Neoplasms', 'Nervous System', 'Perinatal Period', 
       'Pregnancy', 'Respiratory System',
       'Skin and Subcutaneous Tissue', 'Symptoms Conditions', 'None': """)
if g != 'None':
    if '&' in g:
        k = g.split('&')
        for i in k:
            j = i.split(':')
            lst7 = lst7.append({'{}'.format(str(j[0])): int(j[1])}, ignore_index=True)
        lst7 = lst7.max().to_frame().T
    else:
        lst7 = lst7.append({'{}'.format(str(g)): int(1)}, ignore_index=True)
else:
    lst7 = pd.DataFrame(a, columns = col)


lst8 = pd.DataFrame([], columns = ["STOP_D/C'd", 'STOP_NotStopd', 'STOP_Restart', 'STOP_Stopped'])
g = input("Enter STOPPED type (D/C'd, NotStopd, Restart,Stopped): ")
lst8 = lst8.append({'STOP_{}'.format(str(g)): int(1)}, ignore_index=True)


lst9 = pd.DataFrame([], columns = ['EVEN_admit', 'EVEN_discharge', 'EVEN_transfer'])
g = input("Enter EVEN type ('admit', 'discharge', 'transfer'): ")
lst9 = lst9.append({'EVEN_{}'.format(str(g)): int(1)}, ignore_index=True)


cols = ['SER_CMED', 'SER_CSURG', 'SER_MED',
       'SER_NMED', 'SER_NSURG', 'SER_OMED', 'SER_OTHER', 'SER_SURG',
       'SER_TRAUM', 'SER_VSURG']
lst10 = pd.DataFrame([], columns = cols)
g = input("""Enter Service type ('CMED', 'CSURG', 'MED',
       'NMED', 'NSURG', 'OMED', 'OTHER', 'SURG',
       'TRAUM', 'VSURG'): """)
lst10 = lst10.append({'SER_{}'.format(str(g)): int(1)}, ignore_index=True)

lsticu = pd.DataFrame([], columns = ['CARE_ICU'])
g = input('Enter CARE_ICU number: ')
lsticu = lsticu.append({'CARE_ICU': int(g)}, ignore_index=True)  

lst11 = pd.DataFrame([], columns = ['AGE_newborn', 'AGE_teens', 'AGE_young-adult', 'AGE_adult', 'AGE_senior'])
g = input("Enter Age ('newborn', 'teens', 'young-adult', 'adult', 'senior'): ")
lst11 = lst11.append({'AGE_{}'.format(str(g)): int(1)}, ignore_index=True)

lst12 = pd.DataFrame([], columns = ['DB_carevue',
       'DB_carevue & metavision', 'DB_metavision'])
g = input("Enter DB type ('carevue', 'carevue & metavision', 'metavision'): ")
lst12 = lst12.append({'DB_{}'.format(str(g)): int(1)}, ignore_index=True)

lst13 = pd.DataFrame([], columns = ['GENDER_F', 'GENDER_M'])
g = input("Enter Gender ('F', 'M'): ")
lst13 = lst13.append({'GENDER_{}'.format(str(g)): int(1)}, ignore_index=True)

lst14 = pd.DataFrame([], columns = ['INSUR_Government', 'INSUR_Medicaid', 'INSUR_Medicare', 'INSUR_Private',
       'INSUR_Self Pay'])
g = input("Enter Insurance Type ('Government', 'Medicaid', 'Medicare', 'Private', 'Self Pay'): ")
lst14 = lst14.append({'INSUR_{}'.format(str(g)): int(1)}, ignore_index=True)

lst15 = pd.DataFrame([], columns = ['RELI_CATHOLIC', 'RELI_JEWISH', 'RELI_NOT SPECIFIED',
       'RELI_OTHER', 'RELI_PROTESTANT QUAKER', 'RELI_UNOBTAINABLE'])
g = input("Enter Religion Type ('NOT SPECIFIED', 'CATHOLIC', 'JEWISH', 'PROTESTANT QUAKER','UNOBTAINABLE', 'OTHER'): ")
lst15 = lst15.append({'RELI_{}'.format(str(g)): int(1)}, ignore_index=True)

lst16 = pd.DataFrame([], columns = ['MARR_DIVORCED', 'MARR_LIFE PARTNER', 'MARR_MARRIED', 'MARR_OTHER',
       'MARR_SEPARATED', 'MARR_SINGLE', 'MARR_UNKNOWN (DEFAULT)',
       'MARR_WIDOWED'])
g = input("""Enter Marriage ('OTHER', DIVORCED', 'LIFE PARTNER', 
          'MARRIED', 'SEPARATED', 'SINGLE', 'WIDOWED',
          'UNKNOWN (DEFAULT)'): """)
lst16 = lst16.append({'MARR_{}'.format(str(g)): int(1)}, ignore_index=True)

lst17 = pd.DataFrame([], columns = ['ETH_ASIAN', 'ETH_BLACK/AFRICAN AMERICAN', 
       'ETH_HISPANIC OR LATINO', 'ETH_OTHER', 'ETH_UNKNOWN/NOT SPECIFIED',
       'ETH_WHITE'])
g = input("Enter Ethnicity Type (ASIAN,BLACK/AFRICAN AMERICAN,HISPANIC OR LATINO,WHITE, OTHER, UNKNOWN/NOT SPECIFIED): ")
lst17 = lst17.append({'ETH_{}'.format(str(g)): int(1)}, ignore_index=True)

df_test = pd.concat([lst1,lst2,lst4,lst5,lst6,lst7,lst8,lst9,lst10,lsticu,lst11,lst12,lst13,lst14,lst15,lst16,lst17], axis=1)
df_test = df_test.fillna(0)

with open('ui/reg_model', 'rb') as f:
    reg_model = pickle.load(f)

pred = reg_model.predict(df_test)

# By Classification:
with open('ui/class_model', 'rb') as f:
    class_model = pickle.load(f)

pred_class = class_model.predict(df_test)

print('Predict LOS is {} days, Predict LOS class is {}.'.format(round(float(pred)), str(pred_class)))


