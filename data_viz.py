import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt


ALL_TABLES_PATH_ORG = '/USERS/vibhavagarwal/Downloads/tables_redi2_1/'
OUTPUT_DIR = '/USERS/vibhavagarwal/Desktop/table2vec/all_tables'

def read_table(table):
    if table.split('.')[-1] == 'json':
        table = table.split('.')[0]
    with open(os.path.join(OUTPUT_DIR, f"{table}.json"), 'r') as f:
        j = json.load(f)
    return j

def get_stats(all_tables):
    tables = [[],[]]
    for js in all_tables:
        j = read_table(js)
        tables[0].append(j['numDataRows'])
        tables[1].append(j['numCols'])
    tables[0].sort()
    tables[1].sort()
    return tables

if __name__ == "__main__":

    all_tables = os.listdir(OUTPUT_DIR)
    # baseline_f = pd.read_csv('../../www2018-table/feature/features.csv')
    # all_tables =  list(baseline_f['table_id'])

    tables = get_stats(all_tables)
    cols = []
    for i in list(set(tables[1])):
        cols.append(len([k for k in tables[1] if k==i]))
    print(cols)
    rows = []
    for i in list(set(tables[0])):
        rows.append(len([k for k in tables[0] if k==i]))
    print(rows)
    plt.plot(cols)
    plt.show()
    plt.plot(rows)
    plt.show()
