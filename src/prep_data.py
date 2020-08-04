from __future__ import absolute_import, division, print_function
import os
import pickle
import pandas as pd
from collections import Counter
import codecs


def save_pickle(data, filename):
    '''save data as pkl format'''

    filepath = os.path.join(filename)
    f = open(filepath, 'wb')
    pickle.dump(data, f)
    f.close()


def load_pickle(filepath):
    '''load pkl format data'''

    f = open(filepath, 'rb')
    file = pickle.load(f)
    return file


def word_2_idx(words, idx_dict):
    '''convert word to index from dictionary'''

    result = []
    for word in words:
        result.append(idx_dict[word])
    
    return result


def idx_2_word(ids, word_dict):
    '''convert index back to words from dictionary'''

    result = []
    for idx in ids:
        result.append(word_dict[idx])

    return ' '.join(result)

def make_pkl(filepath):
    '''build pattern and intention pkl format file from raw data'''

    pattern_pkl = "../data/pattern.pkl"
    intention_pkl = "../data/intention.pkl"
    total_p = []
    total_i = []
    total_pi = []

    csv_file = pd.read_csv(filepath)
    patterns = list(csv_file["pattern"])
    intentions = list(csv_file["intention"])

    for single_p, single_i in zip(patterns, intentions):
        total_p.append(single_p)
        total_i.append(single_i)
        total_pi.append(single_p+";"+single_i)

    save_pickle(total_p, pattern_pkl)
    save_pickle(total_i, intention_pkl)

    total_pi = list(set(total_pi))
    save_pickle(total_pi, "../data/pattern_intention.pkl")
    print("DATA DONE!")

    return len(total_pi)


def prepare_pkl(filepath):
    '''load pkl data for train'''

    pattern_pkl = "../data/pattern.pkl"
    intention_pkl = "../data/intention.pkl"

    total_pi = []

    f = codecs.open(filepath, encoding='utf8')
    lines = f.readlines()

    for single_line in lines[1:]:
        single_line_items = single_line.split("\t")
        total_pattern.append(single_line_items[0])
        total_intention.append(single_line_items[1])

    # if pkl data already exsits, overwrite new file
    if os.path.exists(pattern_pkl):
        pre_p_list = list(load_pickle(pattern_pkl))
        pre_i_list = list(load_pickle(intention_pkl))

        total_pattern.extend(pre_p_list)
        total_intention.extend(pre_i_list)

        os.remove(pattern_pkl)
        os.remove(intention_pkl)

        save_pickle(total_pattern, pattern_pkl)
        save_pickle(total_intention, intention_pkl)

    else:
        save_pickle(total_pattern, pattern_pkl)
        save_pickle(total_intention, intention_pkl)


    pre_p_list = list(load_pickle(pattern_pkl))
    pre_i_list = list(load_pickle(intention_pkl))

    for i, j in zip(pre_p_list, pre_i_list):
        total_pi.append(';'.join([i, j]))

    save_pickle(list(set(total_pi)), "../data/pattern_intention.pkl")
    print("All done!")

    return len(list(set(total_pi)))


def build_dict():
    pattern_pkl = "../data/pattern_intention.pkl"
    intention_pkl = "../data/intention.pkl"
    #intention = list(load_pickle(intention_pkl))
    p = list(load_pickle(pattern_pkl))

    # intention dictionary
    int_list = [word.split(";")[1] for word in p]
    int_counts = Counter(int_list)
    freq_tag_list = [tag[0] for tag in int_counts.most_common()]

    tag_dictionary = dict([tag, i] for i, tag in enumerate(freq_tag_list))
    tag_inv_dictionary = freq_tag_list
    save_pickle(tag_dictionary, "../data/int_dict.pkl")
    save_pickle(tag_inv_dictionary, "../data/inv_int_dict.pkl")
    tag_size = len(freq_tag_list)

    # pattern dictionary
    patterns = [single_data.split(";")[0] for single_data in p if single_data.split(";")[1] in freq_tag_list]
    whole_text = " ".join(patterns)
    
    word_list = whole_text.split()
    freq_word_list = [word[0] for word in Counter(word_list).most_common()]
    word_dictionary = dict([word, i+2] for i, word in enumerate(freq_word_list))
    
    word_dictionary['_STR_'] = 0
    word_dictionary['_END_'] = 1
    word_inv_dictionary = ["_STR_", "_END_"]
    word_inv_dictionary.extend(freq_word_list)
    
    save_pickle(word_dictionary, "../data/word_dict.pkl")
    save_pickle(word_inv_dictionary, "../data/inv_word_dict.pkl")
    vocab_size = len(freq_word_list)

    os.remove(pattern_pkl)
    os.remove(intention_pkl)

    renew_data = [single_data for single_data in p if single_data.split(";")[1] in freq_tag_list]
    data_length = len(renew_data)

    save_pickle(renew_data, "../data/pattern_intention.pkl")

    return data_length, vocab_size+2, tag_size



def generate_online(idx):
    
    data = load_pickle("../data/pattern_intention.pkl")
    word_dict = load_pickle("../data/word_dict.pkl")
    tag_dict = load_pickle("../data/int_dict.pkl")
    current_data = data[idx].split(";")
    curr_sent = word_2_idx(current_data[0].split(), word_dict)

    curr_input_p = [0]
    curr_input_p.extend(curr_sent)

    curr_target = curr_sent
    curr_target.append(1)

    curr_input_i = word_2_idx([current_data[1]], tag_dict)*len(curr_input_p)

    return curr_input_p, curr_target, curr_input_i, len(curr_input_p)


def prepare_data(filepath):

    data = "../data/pattern_intention.pkl"
    inv_word_dict = "../data/inv_word_dict.pkl"
    inv_tag_dict = "../data/inv_int_dict.pkl"

    if not os.path.exists(data):
        make_pkl(filepath)
        dl, vs, ts = build_dict()

    else:
        vs = len(load_pickle(inv_word_dict))
        ts = len(load_pickle(inv_tag_dict))
        dl = len(load_pickle(data))

    return dl, vs, ts


if __name__ == "__main__":

    a, b, c = prepare_data("../data/sample.csv")
    inv_tag_dict = "../data/inv_int_dict.pkl"
    ts = load_pickle(inv_tag_dict)
    print(ts)
