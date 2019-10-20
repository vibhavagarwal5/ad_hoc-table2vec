import os
import sys
import json
import re
from nltk.corpus import stopwords
stops = stopwords.words('english')


def get_all_strings_from_list(list):
    strings = []
    for obj in list:
        if not isinstance(obj, str):
            strings += get_all_strings_from_list(obj)
        else:
            strings.append(obj)

    return strings

# read data into a list of strings


def read_data(table):
    data = []
    # get words from the what is under the keys ["title", "pgTitle", "secondTitle", "caption", "data"]
    keys = ["title", "pgTitle", "secondTitle", "caption", "data"]

    for key in keys:
        # if table does not have key, skip this key
        if not key in table:
            continue
        strings = table[key]
        data += get_all_strings_from_list([strings])

    # filter data
    data = filter(data)
    return data


def filter(strings):
    data = []

    for string in strings:
        # make every string lowercase
        string = str.lower(string)

        # filter out html tag span style
        #string = re.sub(r'<span.*?/span>', '', string, re.DOTALL)
        string = re.sub(r'<span style.*?/span>', '', string)
        string = re.sub(r'<.*?>', '', string)

        # filter out url
        string = re.sub(r'(?P<url>https?://[^\s]+)', '', string)

        # remove whitespace on the sides
        string = str.strip(string)

        # skip empty words
        if string == '':
            continue
        # skip words that only contain numbers
        if re.match(r'^[0-9]+$', string):
            continue

        # replace symbols with a space
        string = re.sub(r'[^a-z\.]+', ' ', string)

        # remove whitespace on the sides again
        # (the symbol replacement might have added spaces on the sides)
        string = str.strip(string)
        # skip useless strings
        if string == '' or string == '.' or string == 'none' or len(string) < 3 or string in stops:
            continue

        # split on whitespace
        words = string.split(' ')
        data += words
    return data


def parseData(filename, table_name):
    try:
        inp = json.load(open(filename))
        input_data = read_data(inp[table_name])
    except Exception as e:
        print(filename)
        filename_split = filename.split('-')
        filename = filename_split[0] + "-" + \
            str(int(filename_split[1].split('.')[0])-1) + '.json'
        print(filename)
        inp = json.load(open(filename))
        input_data = read_data(inp[table_name])
    data = filter(input_data)
    data = ' '.join(data)

    return data
