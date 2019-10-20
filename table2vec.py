import pandas as pd
import tensorflow as tf
import numpy as np
import gensim
import gensim.models.keyedvectors as word2vec
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import ast
from pandarallel import pandarallel

from parse_data import parseData
from get_entities import getEntities

pandarallel.initialize(progress_bar=True, nb_workers=15, shm_size_mb=2500)

w2v_gn_model = word2vec.KeyedVectors.load_word2vec_format(
    "./GoogleNews-vectors-negative300.bin", binary=True)
rdv2vec_model = gensim.models.Word2Vec.load('./DB2Vec_sg_200_5_5_15_2_500')

tables_path = '/home/vibhav/table2vec/tables_redi2_1/'
def get_table_path(table_id):
    return tables_path + 're_tables-' + table_id.split('-')[1] + '.json'


def preprocess_table(baseline_f):
    baseline_f['table_path'] = baseline_f.table_id.apply(get_table_path)
    return baseline_f


def get_w2v_embd(sentence):
    embd = []
    for word in sentence:
        try:
            embd.append(w2v_gn_model.get_vector(word))
        except Exception as e:
            embd.append(w2v_gn_model.get_vector('UNK'))
    embd = np.array(embd)
    return embd


def get_g2v_embd(entities):
    embd = []
    for en in entities:
        word = 'dbr:' + en
        try:
            embd.append(rdv2vec_model.wv[word])
        except Exception as e:
            embd.append(rdv2vec_model.wv['dbr:UNK'])
    return np.array(embd)


def early_fusion(table, query):
    a = np.average(table, axis=0).reshape(1, -1)
    b = np.average(query, axis=0).reshape(1, -1)
    sim = cosine_similarity(a, b)
    return sim.reshape(-1)[0]


def late_fusion(table, query):
    s = []
    for i in query:
        for j in table:
            sim = cosine_similarity(i.reshape(1, -1), j.reshape(1, -1))
            s.append(sim)
    s = np.array(s).reshape(-1)
    return s


def get_late_fusion_scores(semantic_f, embd_type):
    if embd_type == 'w':
        semantic_f['w2v_late_fusion'] = semantic_f.parallel_apply(
            lambda x: late_fusion(x['w2v_embd_table'], x['w2v_embd_query']), axis=1)
        semantic_f['w2v_late_fusion_max'] = semantic_f.w2v_late_fusion.apply(
            np.max)
        semantic_f['w2v_late_fusion_avg'] = semantic_f.w2v_late_fusion.apply(
            np.average)
        semantic_f['w2v_late_fusion_sum'] = semantic_f.w2v_late_fusion.apply(
            np.sum)
    elif embd_type == 'g':
        semantic_f['g2v_late_fusion'] = semantic_f.parallel_apply(
            lambda x: late_fusion(x['g2v_embd_table'], x['g2v_embd_query']), axis=1)
        semantic_f['g2v_late_fusion_max'] = semantic_f.g2v_late_fusion.apply(
            np.max)
        semantic_f['g2v_late_fusion_avg'] = semantic_f.g2v_late_fusion.apply(
            np.average)
        semantic_f['g2v_late_fusion_sum'] = semantic_f.g2v_late_fusion.apply(
            np.sum)
    return semantic_f


def get_query_entities(semantic_f):
    query_entities = {}
    for i in semantic_f['query'].unique():
        query_entities[i] = getEntities(i)
    semantic_f['query_entites'] = semantic_f['query'].apply(
        lambda x: query_entities[x])
    return semantic_f


if __name__ == "__main__":
    baseline_f = pd.read_csv('./www2018-table/feature/features.csv')
    baseline_f = preprocess_table(baseline_f)
    semantic_f = baseline_f.loc[:, ['query_id',
                                    'query', 'table_path', 'table_id', 'rel']]

    # Word Embeddings
    semantic_f['parsedTable'] = semantic_f.parallel_apply(
        lambda x: parseData(x['table_path'], x['table_id']).split(' '), axis=1)
    semantic_f['w2v_embd_table'] = semantic_f.parsedTable.apply(get_w2v_embd)
    semantic_f['w2v_embd_query'] = semantic_f['query'].apply(
        lambda x: get_w2v_embd(x.split(" ")))

    semantic_f['w2v_early_fusion'] = semantic_f.apply(
        lambda x: early_fusion(x['w2v_embd_table'], x['w2v_embd_query']), axis=1)
    semantic_f = get_late_fusion_scores(semantic_f)

    # Graph Embeddings
    semantic_f = get_query_entities(semantic_f)

    # From here onwards, we had all the above except table_entites.
    df_1 = pd.read_csv('./semantic_f_w2v_entities_1.csv')
    df_2 = pd.read_csv('./semantic_f_w2v_entities_2.csv')
    df_3 = pd.read_csv('./semantic_f_w2v_entities_3.csv')
    df_4 = pd.read_csv('./semantic_f_w2v_entities_4.csv')
    df = pd.concat([df_1, df_2, df_3, df_4]) 
    # df is same as semantic_f now
    
    df['table_entities'] = df.table_entities.apply(
        lambda x: ast.literal_eval(x))

    df['g2v_embd_table'] = df.table_entities.apply(get_g2v_embd)
    df['g2v_embd_query'] = df.query_entites.apply(get_g2v_embd)

    df['g2v_early_fusion'] = df.apply(lambda x: early_fusion(
        x['g2v_embd_table'], x['g2v_embd_query']), axis=1)
    df = get_late_fusion_scores(df)
