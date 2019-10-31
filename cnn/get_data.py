import numpy as np
import pandas as pd
import os
import json
from multiprocessing import Pool
import math
import random
import pickle

ALL_TABLES_PATH_ORG = '/USERS/vibhavagarwal/Downloads/tables_redi2_1/'
OUTPUT_DIR = '/USERS/vibhavagarwal/Desktop/table2vec/all_tables'
MAX_COL_LEN = 20
SAMPLE_PERC_ROW = 0.2

# def read_table(filename, table_name):
#     try:
#         inp = json.load(open(filename))[table_name]
#     except Exception as e:
#         print(filename)
#         filename_split = filename.split('-')
#         filename = filename_split[0] + "-" + \
#             str(int(filename_split[1].split('.')[0])-1) + '.json'
#         print(filename)
#         inp = json.load(open(filename))[table_name]
#     return inp

# def get_table_path(table_id):
#     return ALL_TABLES_PATH_ORG + 're_tables-' + table_id.split('-')[1] + '.json'


def create_table(js):
    print(js)
    with open(os.path.join(ALL_TABLES_PATH_ORG, js), 'r') as f:
        j = json.load(f)
        for table in j.keys():
            out = clean_data(j[table]['data'])
            j[table]['data'] = out
            with open(os.path.join(OUTPUT_DIR, f"{table}.json"), 'w') as f:
                json.dump(j[table], f)


def clean_data(table_d):
    for row_id, row in enumerate(table_d):
        for col_id, col in enumerate(row):
            table_d[row_id][col_id] = filter_d(table_d[row_id][col_id])
    return table_d


def filter_d(inp):
    if len(inp):
        if inp[0] == '[' and inp[-1] == ']':
            return inp.split('|')[0][1:]
        else:
            inp = inp.strip().split(" ")
            inp = '_'.join(inp)
            inp = (inp.encode('ascii', 'ignore')).decode("utf-8")
            inp = inp.strip()
            return inp
    else:
        return inp


def delete_json(js):
    print(js)
    os.remove(os.path.join(OUTPUT_DIR, js))


def create_dataset(table):
    print(table)
    if table.split('.')[-1] == 'json':
        table = table.split('.')[0]
        
    X = []
    y = []
    with open(os.path.join(OUTPUT_DIR, f"{table}.json"), 'r') as f:
        j = json.load(f)
    if j['numCols'] == 0 or j['numDataRows'] == 0:
        return X, y
    for row in j['data']:
        if len(row) > MAX_COL_LEN:
            print("Splitting the row")
            splits = split_row(row)
            for v in splits:
                X.append(v)
                y.append(1)
                neg_sample = generate_neg(j['data'], v)
                X.append(neg_sample)
                y.append(-1)
        else:
            X.append(row)
            y.append(1)
            neg_sample = generate_neg(j['data'], row)
            X.append(neg_sample)
            y.append(-1)
    return X, y


def split_row(row):
    length = len(row)
    r = math.ceil(length/MAX_COL_LEN)
    for i in range(0, r-1, MAX_COL_LEN):
        yield row[i:i+MAX_COL_LEN]


def generate_rand_table(table, no_sample):
    rand_table = random.choice(all_tables)
    if rand_table.split('.')[-1] == 'json':
        rand_table = rand_table.split('.')[0]
    print('Getting the random table')
    with open(os.path.join(OUTPUT_DIR, f"{rand_table}.json"), 'r') as f:
        j = json.load(f)
    while j['data'] == table or j['numDataRows']*j['numCols'] <= no_sample:
        print('sample table again')
        j = generate_rand_table(table, no_sample)
    return j


def generate_neg(table, row):
    no_sample = math.ceil(SAMPLE_PERC_ROW * len(row))
    a = list(range(len(row)))
    random.shuffle(a)
    random_ixs = a[:no_sample]

    j = generate_rand_table(table, no_sample)

    print(
        f"no of samples: {no_sample}, sampled table rows: {j['numDataRows']}, sampled table columns: {j['numCols']}")
    print(
        f"input table rows: {len(table)}, input table columns: {len(table[0])}")

    print('Getting the random value from the table')
    rand_row_ix = random.choice(list(range(j['numDataRows'])))
    rand_col_ix = random.choice(list(range(j['numCols'])))
    rand_val = j['data'][rand_row_ix][rand_col_ix]
    c = 0
    while rand_val in (text for rw in table for text in rw) and c <= no_sample:
        print('sample value again')
        print(rand_row_ix, rand_col_ix, j['data']
              [rand_row_ix][rand_col_ix], rand_val)
        # print([text for rw in table for text in rw])
        rand_row_ix = random.choice(list(range(j['numDataRows'])))
        rand_col_ix = random.choice(list(range(j['numCols'])))
        rand_val = j['data'][rand_row_ix][rand_col_ix]
        c += 1
    if c > no_sample:
        return generate_neg(table, row)
    else:
        for ix in random_ixs:
            row[ix] = rand_val
        return row


# Testing tables
# table-0614-640.json
# table-1225-209.json

if __name__ == "__main__":

    # all_tables = os.listdir(OUTPUT_DIR)
    baseline_f = pd.read_csv('../../www2018-table/feature/features.csv')
    all_tables = list(baseline_f['table_id'])

    # p = Pool(processes=30)
    # result = p.map(create_table, all_json)
    # result = p.map(delete_json, all_json)
    X = []
    y = []
    for table in all_tables:
        X_, y_ = create_dataset(table)
        X.append(X_)
        y.append(y_)
    with open('X_l20.pkl', 'wb') as f:
        pickle.dump(X, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('y_l20.pkl', 'wb') as f:
        pickle.dump(y, f, protocol=pickle.HIGHEST_PROTOCOL)
