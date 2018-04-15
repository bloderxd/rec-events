import pandas as pd
import os
from sklearn import preprocessing


def load_fixture():
    df = pd.read_csv(os.path.dirname(__file__) + '/users_regs.csv', sep=',',
                names=['user', 'item', 'visits', 'category'], header=None)
    rating = df['visits'].values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(rating.reshape(-1, 1))
    df_normalized = pd.DataFrame(x_scaled)
    df['visits'] = df_normalized
    return df
