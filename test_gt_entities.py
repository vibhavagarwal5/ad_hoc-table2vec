import requests
import os
import sys
import subprocess
import json
import time
import numpy as np
import pandas as pd
from multiprocessing import  Pool
import logging

# from loguru import logger
# logger.add("stdout_4.log", enqueue=True)

class StreamToLogger(object):
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''
    
    def flush(self):
        pass

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

logging.basicConfig(
   level=logging.DEBUG,
   format='%(asctime)s:%(levelname)s:%(message)s',
   filename="stdout_4.log",
   filemode='a'
)

stdout_logger = logging.getLogger('STDOUT')
sl = StreamToLogger(stdout_logger, logging.INFO)
sys.stdout = sl

stderr_logger = logging.getLogger('STDERR')
sl = StreamToLogger(stderr_logger, logging.ERROR)
sys.stderr = sl

def getEntities(query = '',file_path = '',table_name = ''):
    if query != '':
        # url = 'http://api.nordlys.cc/er?q=' + '+'.join(query.split(" ")) + '&1st_num_docs=' + str(output_count) + '&model=mlm'
        # r = requests.get(url)
        # entities = r.json()
        
        entities = entities_request(query,'er')
        return entities['results']
    else:
        table = getTable(file_path, table_name)
        title_entities = from_meta_data(table,'pgTitle')
        caption_entities = from_meta_data(table,'caption')
        df = pd.DataFrame(np.array(table['data']))
        entities = {}
        for col in df.columns:
            entities[col] = from_columns(df[col])
        max_len = 0 
        i = 0
        for key in entities.keys():
            if max_len < len(entities[key].values()):
                max_len = len(entities[key].values())
                i = key
        core_col_entities = list(entities[i].values())
        print(core_col_entities)
        all_entities = list(set().union(*[title_entities, caption_entities, core_col_entities]))
        print(all_entities)
        return all_entities

def entities_request(inp,req_type):
    print('Before: ' + inp)
    item = preprocess_word(inp)
    print('After: ' + item)
    os.chdir('/home/vibhav/nordlys/')
    if req_type == 'er':
        cmd = 'python -m nordlys.services.er -c config.json -q "' + str(item) + '"'
    else:
        cmd = 'python -m nordlys.services.el -q "' + str(item) + '"'
    result = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout.read()
    print(cmd +" CMD:RESULT "+ str(result))
    return eval(result.decode('utf-8'))

def from_meta_data(table,extra_type):
    # entities = requests.get('http://api.nordlys.cc/er?q=' + '+'.join(table[extra_type].split(" ")) + '&1st_num_docs=' + str(output_count) + '&model=mlm').json()
    entities = entities_request(table[extra_type],'er')
    data = [entities['results'][key]['entity'][1:-1].split(":")[-1] for key in entities['results'].keys()]
    print(data)
    return data
    
def from_columns(col):
    print("Before",col)
    col = list(set(col))
    print("After",col)
    entities_col = {}
    for item in col:
        if len(item.split("|")) == 2:
            entities_col[item] = item.split("|")[0][1:]
        else:
            # url = 'http://api.nordlys.cc/el?q=' + '+'.join(item.split(" "))
            # r = requests.get(url)
            # data = r.json()
            # data = data['results']
            entities = entities_request(item,'el')
            data = entities['results']
            print(item +" ITEM:DATA "+ str(data))
            if len(data)!=0:
                data.sort(key = lambda x: x['score'], reverse = True)
                entities_col[item] = data[0]['entity'][1:-1].split(":")[-1]
    return entities_col

def preprocess_word(item):
    if len(item) == 0:
        item = ' '
    if '`' in item:
        item = "'".join(item.split('`'))
    if '+' in item:
        item = " ".join(item.split("+"))
    if '"' in item:
        item = item.replace("\"","\'")
    if len(item) == 0:
        item = ' '
    elif item[0] in ['-','$'] or item[-1] == '\\':
            item = ' ' + item + ' '
    return item

def getTable(file_path, table_name):
    try:
        inp = json.load(open(file_path))
        table = inp[table_name]
    except Exception as e:
        file_path_split = file_path.split('-')
        file_path = file_path_split[0] + "-" + str(int(file_path_split[1].split('.')[0])-1) + '.json'
        inp = json.load(open(file_path))
        table = inp[table_name]
    return table

df = pd.read_csv('./semantic_f_w2v.csv')

# file_path = './tables_redi2_1/re_tables-1000.json'
# table_name = 'table-1000-57'
# file_path = './tables_redi2_1/re_tables-0255.json'
# table_name = 'table-0255-236'
# table_name = 'table-0255-586'

# start_time = time.time()
# print(getEntities(file_path = file_path, table_name = table_name))
# print(time.time()-start_time)

def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

df['table_path'] = df['table_path'].apply(lambda x: '/home/vibhav/table2vec' + x[1:])

def find_ent(df):
    df['table_entities'] = df.apply(lambda x: getEntities(file_path = x['table_path'],table_name = x['table_id']),axis=1)
    return df

df = parallelize_dataframe(df.iloc[2500:], find_ent, 15)

df.to_csv('/home/vibhav/table2vec/semantic_f_w2v_entities_4.csv',index=False)
