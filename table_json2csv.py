import numpy as np
import pandas as pd
import json
import csv

path = '/home/vibhav/table2vec/tables_redi2_1_csv/'

def json2csv(file_path, table_name):
	try:
		inp = json.load(open(file_path))
		data = inp[table_name]['data']
	except Exception as e:
		filename_split = file_path.split('-')
		file_path = filename_split[0] + "-" + str(int(filename_split[1].split('.')[0])-1) + '.json'
		inp = json.load(open(file_path))
		data = inp[table_name]['data']
	writer = csv.writer(open(path + table_name + ".csv", 'w+'))
	# s = ''
	for row in data:
		writer.writerow(row)
		# for item in row:
		#     s = s + item + ','
		# s = s[:-1] + '\n'

