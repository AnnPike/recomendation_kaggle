import pandas as pd
from pathlib import Path
import os
import random
import numpy as np
import json
from datetime import timedelta
from collections import Counter
from heapq import nlargest
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import gc
import warnings
warnings.filterwarnings('ignore')

### Paths ###

DATA_PATH = Path('otto-recommender-system')
TRAIN_PATH = DATA_PATH/'train.jsonl'
TEST_PATH = DATA_PATH/'test.jsonl'
SAMPLE_SUB_PATH = Path('otto-recommender-system/sample_submission.csv')


def rate(x):
    if x == 'clicks':
        return 1
    elif x == 'carts':
        return 2
    elif x == 'orders':
        return 3


min_events_per_session = 50
sample_size = 150000

chunks = pd.read_json(TEST_PATH, lines=True, chunksize=sample_size) #try for test, change to train later!

df_session_aid = pd.DataFrame(dtype='uint8')
session = 0
for c in tqdm(chunks):
    gc.collect()
    sample_train_df = c
    sample_train_df.set_index('session', drop=True, inplace=True)
    for i in range(sample_train_df.shape[0]):
        session = session + 1
        df_event = pd.DataFrame(sample_train_df.iloc[i].item())
        if len(df_event) >= min_events_per_session:
            df_event['type'] = df_event['type'].apply(rate)
            df_event_rated = df_event.groupby('aid')['type'].max()
            df_event_rated.name = session
            df_session_aid = df_session_aid.append(df_event_rated)
    print('\n', df_session_aid.tail(), '\n', df_session_aid.shape)
    pickle.dump(df_session_aid, open('session_aid_df', 'wb'))