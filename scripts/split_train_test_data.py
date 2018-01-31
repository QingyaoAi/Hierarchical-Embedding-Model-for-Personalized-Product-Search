import os,sys
import gzip
import random
import numpy as np
import json

data_path = sys.argv[1]
review_sample_rate = float(sys.argv[2])
query_sample_rate = float(sys.argv[3])
output_path = data_path + 'query_split/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

#read all queries
product_query_idxs = []
query_level_count = []
all_queries = {}
all_query_idx = []
with gzip.open(data_path + 'product_query.txt.gz', 'r') as cat_fin:
	for line in cat_fin:
		query_arr = line.strip().split(';')
		product_query_idxs.append([])
		for query_line in query_arr:
			if len(query_line) < 1:
				continue
			arr = query_line.split('\t')
			cat_num = int(arr[0][1:])
			cat_query = arr[1]
			if cat_query not in all_queries:
				all_queries[cat_query] = len(all_queries)
				all_query_idx.append(cat_query)
				query_level_count.append(cat_num)
			product_query_idxs[-1].append(all_queries[cat_query])

#build user-review map
user_review_map = {}
review_product_list = []
with gzip.open(data_path + 'review_u_p.txt.gz', 'r') as fin:
	index = 0
	for line in fin:
		arr = line.strip().split(' ')
		user = arr[0]
		product = int(arr[1])
		if user not in user_review_map:
			user_review_map[user] = []
		user_review_map[user].append(index)
		review_product_list.append(product)
		index += 1

#generate train/test sets
test_review_idx = set()
for user in user_review_map:
	sample_number = int(len(user_review_map[user]) * review_sample_rate)
	test_review_idx |= set(np.random.choice(user_review_map[user], sample_number , replace=False))

test_query_idx = set()
'''	#sample per product: easily to sample high frequency queries as test
test_product = set()
for review_idx in test_review_idx:
	product_idx = review_product_list[review_idx]
	if product_idx not in test_product:
		sample_number = int(len(product_query_idxs[product_idx]) * query_sample_rate)
		potential_queries = set(np.random.choice(product_query_idxs[product_idx], sample_number , replace=False))		
		for i in potential_queries:
			if query_level_count[i] >= 2:
				test_query_idx.add(i)
'''
sample_number = int(len(all_query_idx) * query_sample_rate)
test_query_idx = set(np.random.choice([i for i in xrange(len(all_query_idx))], sample_number , replace=False))		
# refine the train query set so that every item has at least one query
for query_idxs in product_query_idxs:
	tmp = set(query_idxs) - test_query_idx
	if len(tmp) < 1:
		pick_i = int(random.random()*len(query_idxs))
		test_query_idx.remove(query_idxs[pick_i])


#output train/test review data
train_user_product_map = {}
test_user_product_map = {}
with gzip.open(output_path + 'train.txt.gz', 'w') as train_fout, gzip.open(output_path + 'test.txt.gz', 'w') as test_fout:
	with gzip.open(data_path + 'review_u_p.txt.gz', 'r') as info_fin, gzip.open(data_path + 'review_text.txt.gz', 'r') as text_fin:
		info_line = info_fin.readline()
		text_line = text_fin.readline()
		index = 0
		while info_line:
			arr = info_line.strip().split(' ')
			if index not in test_review_idx:
				train_fout.write(arr[0] + '\t' + arr[1] + '\t' + text_line.strip() + '\n')
				if int(arr[0]) not in train_user_product_map:
					train_user_product_map[int(arr[0])] = set()
				train_user_product_map[int(arr[0])].add(int(arr[1]))
			else:
				test_fout.write(arr[0] + '\t' + arr[1] + '\t' + text_line.strip() + '\n')
				if int(arr[0]) not in test_user_product_map:
					test_user_product_map[int(arr[0])] = set()
				test_user_product_map[int(arr[0])].add(int(arr[1]))
			index += 1
			info_line = info_fin.readline()
			text_line = text_fin.readline()

#read review_u_p and construct train/test id sets
with gzip.open(output_path + 'train_id.txt.gz', 'w') as train_fout, gzip.open(output_path + 'test_id.txt.gz', 'w') as test_fout:
	with gzip.open(data_path + 'review_u_p.txt.gz', 'r') as info_fin, gzip.open(data_path + 'review_id.txt.gz', 'r') as id_fin:
		info_line = info_fin.readline()
		id_line = id_fin.readline()
		index = 0
		while info_line:
			arr = info_line.strip().split(' ')
			if index not in test_review_idx:
				train_fout.write(arr[0] + '\t' + arr[1] + '\t' + str(id_line.strip()) + '\n')
			else:
				test_fout.write(arr[0] + '\t' + arr[1] + '\t' + str(id_line.strip()) + '\n')
			index += 1
			info_line = info_fin.readline()
			id_line = id_fin.readline()

#output train/test queries
query_max_length = 0
with gzip.open(output_path + 'query.txt.gz', 'w' ) as query_fout:
	for cat_query in all_query_idx:
		query_fout.write(cat_query + '\n')
		query_length = len(cat_query.strip().split(' '))
		if query_length > query_max_length:
			query_max_length = query_length

train_product_query_idxs = []
test_product_query_idxs = []
with gzip.open(output_path + 'train_query_idx.txt.gz', 'w') as train_fout, gzip.open(output_path + 'test_query_idx.txt.gz','w') as test_fout:
	for query_idxs in product_query_idxs:
		train_product_query_idxs.append([])
		test_product_query_idxs.append([])
		for q_idx in query_idxs:
			if q_idx not in test_query_idx:
				train_fout.write(str(q_idx) + ' ')
				train_product_query_idxs[-1].append(q_idx)
			else:
				test_fout.write(str(q_idx) + ' ')
				test_product_query_idxs[-1].append(q_idx)
		train_fout.write('\n')
		test_fout.write('\n')

#generate qrels and json queries
product_ids = []
with gzip.open(data_path + 'product.txt.gz', 'r') as fin:
	for line in fin:
		product_ids.append(line.strip())

user_ids = []
with gzip.open(data_path + 'users.txt.gz', 'r') as fin:
	for line in fin:
		user_ids.append(line.strip())

vocab = []
with gzip.open(data_path + 'vocab.txt.gz', 'r') as fin:
	for line in fin:
		vocab.append(line.strip())

def output_qrels_jsonQuery(user_product_map, product_query, qrel_file, jsonQuery_file):
	json_queries = []
	appeared_qrels = {}
	with open(qrel_file, 'w') as fout:
		for u_idx in user_product_map:
			user_id = user_ids[u_idx]
			if user_id not in appeared_qrels:
				appeared_qrels[user_id] = {}
			for product_idx in user_product_map[u_idx]:
				product_id = product_ids[product_idx]
				if product_id not in appeared_qrels[user_id]:
					appeared_qrels[user_id][product_id] = set()
				#check if has query
				for q_idx in product_query[product_idx]:
					if q_idx in appeared_qrels[user_id][product_id]:
						continue
					appeared_qrels[user_id][product_id].add(q_idx)
					fout.write(user_id + '_' + str(q_idx) + ' 0 ' + product_id + ' 1 ' + '\n')
					json_q = {'number' : user_id + '_' + str(q_idx), 'text' : []}
					json_q['text'].append('#combine(')
					for v_i in all_query_idx[q_idx].strip().split(' '):
						if len(v_i) > 0:
							json_q['text'].append(vocab[int(v_i)])
					json_q['text'].append(')')
					json_q['text'] = ' '.join(json_q['text'])
					json_queries.append(json_q)
	with open(jsonQuery_file,'w') as fout:
		output_json = {'mu' : 1000, 'queries' : json_queries}
		json.dump(output_json, fout, sort_keys = True, indent = 4)

output_qrels_jsonQuery(test_user_product_map, test_product_query_idxs, 
				output_path + 'test.qrels', output_path + 'test_query.json')
output_qrels_jsonQuery(train_user_product_map, train_product_query_idxs, 
				output_path + 'train.qrels', output_path + 'train_query.json')

# output statisitc
with open(output_path + 'statistic.txt', 'w') as fout:
	fout.write('Total User ' + str(len(user_ids)) + '\n')
	fout.write('Total Product ' + str(len(product_ids)) + '\n')
	fout.write('Total Review ' + str(len(review_product_list)) + '\n')
	fout.write('Total Vocab ' + str(len(vocab)) + '\n')
	fout.write('Total Queries ' + str(len(all_query_idx)) + '\n')
	fout.write('Max Query Length ' + str(query_max_length) + '\n')
	fout.write('Test Review ' + str(len(test_review_idx)) + '\n')
	fout.write('Test Unique Queries ' + str(len(test_query_idx)) + '\n')


