from __future__ import absolute_import, print_function, division
import os
import sys
import re
import datetime
import shutil
import numpy as np
import tensorflow as tf
import argparse

from prep_data import load_pickle, prepare_data, idx_2_word
from conditional_rnn import C_RNN

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("-t", "--train", type=str, default="train", help="train or inference")
parser.add_argument("-d", "--data", type=str, required=True, help="data path")

## for training
parser.add_argument("-nc", "--num_cells", type=int, default=4, help="number of rnn cells")
parser.add_argument("-nd", "--num_hidden", type=int, default=256, help="number of hidden nodes")
parser.add_argument("-bs", "--batch_size", type=int, default=1, help="training batch size")
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="learning_rate")
parser.add_argument("-ld", "--logdir", type=str, default="../log", help="directory for log")
parser.add_argument("-ep", "--num_epoch", type=int, default=80, help="number of epoch")
parser.add_argument("-ve", "--vocab_emb", type=int, default=256, help="embedding dim for pattern vocab")
parser.add_argument("-te", "--intent_emb", type=int, default=32, help="embedding dim for intention tag")

## for inference
parser.add_argument("-in", "--intention", type=str, default="SEARCH", help="target intention for inference")
parser.add_argument("-mi", "--max_inference", type=int, default=32, help="maximum sentence length for inference")
parser.add_argument("-pr", "--prob", type=float, default=0.08, help="lower bound for next word probability")
parser.add_argument("-np", "--num_pattern", type=int, default=5, help="number of sentences to infer from given intention")
parser.add_argument("-op", "--outpath", type=str, default="./out.txt", help="path and file name for inference")

## for data preparation
parser.add_argument("-re", "--renew", type=int, default=0, help="renew existing data")

parameters = parser.parse_args()

parameters.threshold = parameters.prob


if parameters.train == 'train':
	if parameters.renew == 1:
		if os.path.exists("../data"):
			shutil.rmtree("../data")

	elif parameters.renew == 2:
		if os.path.exists(parameters.logdir):
			shutil.rmtree(parameters.logdir)

	elif parameters.renew == 3:
		if os.path.exists("../data"):
			shutil.rmtree("../data")
		if os.path.exists(parameters.logdir):
			shutil.rmtree(parameters.logdir)

	elif parameters.renew == 0:
		if os.path.exists("../data"):
			answer = input("Overwrite existing data? (y/n)")
			if answer.lower() == "y":
				shutil.rmtree("../data")
				print("Deleted.")
			else:
				print("Keep data.")
		if os.path.exists(parameters.logdir):
			answer = input("Overwrite existing train log? (y/n)")
			if answer.lower() == "y":
				shutil.rmtree(parameters.logdir)
				print("Deleted.")
			else:
				print("Keep train log.")

if not os.path.exists(parameters.logdir):
	os.mkdir(parameters.logdir)
if not os.path.exists("../data"):
	os.mkdir("../data")

data_size, vocab_size, tag_size = prepare_data(parameters.data)
print(data_size, vocab_size, tag_size)

parameters.vocab_size = vocab_size
parameters.tag_size = tag_size
parameters.total_steps = data_size


model = C_RNN(parameters)
print("Graph Completed!")

# train rnn model
if parameters.train == 'train':
	print("Training Start")
	model.train()

# infer rnn model for given intention
elif parameters.train == 'infer':
	tag_dict = load_pickle("../data/int_dict.pkl")
	inv_word_dict = load_pickle("../data/inv_word_dict.pkl")
	print("Retrieving from checkpoint...")
	result = model.infer(intention=tag_dict[parameters.intention])
	print("\nTarget Intention: ", parameters.intention)
	f = open(parameters.filepath, 'w')
	for sent in result:
		#print(sent)
		sentence = idx_2_word(sent[:-1], inv_word_dict)
		print("Generated Pattern: ", sentence)
		f.write(sentence+"\n")
	f.close()

# infer sentence for every intention
else:
	inv_tag_dict = load_pickle("../data/inv_int_dict.pkl")
	inv_word_dict = load_pickle("../data/inv_word_dict.pkl")
	print("Retrieving from checkpoint...")
	f = open(parameters.outpath, "w")
	for i in range(tag_size):
		result = model.infer(intention=i)
		f.write("Input Intention : {}\n".format(inv_tag_dict[i]))
		#print(result)
		for i, sent in enumerate(result):
			#print(sent)
			sentence = idx_2_word(sent[:-1], inv_word_dict)
			f.write("Generated Pattern {:d}: {}\n".format(i+1, sentence))
	f.close()
	print("Text file generated")




