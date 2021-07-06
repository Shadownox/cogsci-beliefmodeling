import pandas as pd
import numpy as np
import ccobra
import os.path

# Check if Trippas dataset exists
if not os.path.isfile('raw_Trippas2018/trippas_drh_syll.txt'):
    print("Please download the Trippas-2018 dataset 'trippas_drh_syll.txt' from https://osf.io/kt3jn/ and copy it into the Trippas2018 subfolder")
    exit()

# Load the raw data
df = pd.read_csv('raw_Trippas2018/trippas_drh_syll.txt', sep='\t')

# Normalize dataset
df['enc_task'] = df['syllc'].apply(lambda x: x[:3].upper())

def extract_enc_resp(x):
    x = x[-2:]
    enc_resp = x[0].upper() + ('ac' if int(x[-1]) == 1 else 'ca')
    assert enc_resp in ccobra.syllogistic.RESPONSES
    return enc_resp

df['enc_resp'] = df['syllc'].apply(extract_enc_resp)
df['response'] = df['rsp'].apply(lambda x: x == 1)
df['is_believable'] = df['bel'].apply(lambda x: x == 'believable')
df['is_valid'] = df['val'].apply(lambda x: x == 'valid')

# Remove uninteresting columns
df = df.drop(columns=[
    'syllc',
    'val',
    'bel',
    'rsp',
])

df = df.rename(columns={
    'sub': 'id',
    'cond': 'experiment_id',
    'conddesc': 'experiment_description',
})

# Add sequence column
seq_nos = []
cur_nos = {}
for row_idx, row in df.iterrows():
    cur_token = (row['id'], row['experiment_id'])
    if cur_token not in cur_nos:
        cur_nos[cur_token] = 0

    seq_nos.append(cur_nos[cur_token])
    cur_nos[cur_token] += 1

df['sequence'] = seq_nos

# Add task
def construct_task(enc_task):
    quant1 = enc_task[0].replace('A', 'All').replace('I', 'Some').replace('E', 'No').replace('O', 'Some not')
    quant2 = enc_task[1].replace('A', 'All').replace('I', 'Some').replace('E', 'No').replace('O', 'Some not')

    if int(enc_task[-1]) == 1:
        return '{};A;B/{};B;C'.format(quant1, quant2)
    elif int(enc_task[-1]) == 2:
        return '{};B;A/{};C;B'.format(quant1, quant2)
    elif int(enc_task[-1]) == 3:
        return '{};A;B/{};C;B'.format(quant1, quant2)
    elif int(enc_task[-1]) == 4:
        return '{};B;A/{};B;C'.format(quant1, quant2)

df['task'] = df['enc_task'].apply(construct_task)

def construct_choice(enc_resp):
    quant = enc_resp[0].replace('A', 'All').replace('I', 'Some').replace('E', 'No').replace('O', 'Some not')

    if enc_resp[1:] == 'ac':
        return '{};A;C'.format(quant)
    else:
        return '{};C;A'.format(quant)

df['choices'] = df['enc_resp'].apply(construct_choice)
df['domain'] = 'syllogistic-belief'
df['response_type'] = 'verify'

# Sort columns
df = df[[
    'id',
    'sequence',
    'task',
    'choices',
    'response',
    'domain',
    'response_type',
    'rating',
    'experiment_id',
    'experiment_description',
    'enc_task',
    'enc_resp',
    'is_believable',
    'is_valid',
]]

print(df.head())
df.to_csv('Trippas2018.csv', index=False)

# Store rating variant of the dataset
df = df.drop(columns=['response'])
df = df.rename(columns={'rating': 'response'})
df['response_type'] = 'single-choice'
df['task'] = df[['task', 'choices']].apply(lambda x: x[0] + '|' + x[1], axis=1)
df['choices'] = '1|2|3|4|5|6'

print(df.head())
df.to_csv('Trippas2018-rating.csv', index=False)
