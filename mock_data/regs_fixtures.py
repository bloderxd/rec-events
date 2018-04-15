import pandas as pd
import os

def load_fixture():
    pd.read_csv(os.path.dirname(__file__) + '/users_regs.csv')


if __name__ == '__main__':
    load_fixture()
