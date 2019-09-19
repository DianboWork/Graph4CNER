import numpy as np
from tqdm import tqdm


def normalize_word(char_list):
    new_list = []
    for char in char_list:
        if char.isdigit():
            new_list.append('0')
        else:
            new_list.append(char)
    return new_list


def get_label(line):
    line_list = line.split(" ")
    label = []
    for ele in line_list:
        if len(ele) == 1:
            label.append("S-SEG")
        elif len(ele) == 2:
            label = label + ["B-SEG", "E-SEG"]
        elif len(ele) >= 3:
            a = ["M-SEG"] * len(ele)
            a[0] = "B-SEG"
            a[-1] = 'E-SEG'
            label = label + a
        else:
            pass
    return label


def read_instance(input_file, gaz, char_alphabet, label_alphabet, gaz_alphabet, number_normalized, max_sent_length):
    in_lines = open(input_file, 'r').readlines()
    instance_ids = []
    instance_texts = []
    cut_num = 0
    for idx in tqdm(range(len(in_lines))):
        line = in_lines[idx].rstrip()
        sent = [ele for ele in line if ele != " "]
        if number_normalized:
            sent = normalize_word(sent)
        char_id = [char_alphabet.get_index(ele) for ele in sent]
        if len(char_id) > 0:
            label = get_label(line)
            label_id = [label_alphabet.get_index(ele) for ele in label]
            assert len(char_id) == len(label)
            if len(sent) >= max_sent_length:
                sent = sent[:max_sent_length]
                char_id = char_id[:max_sent_length]
                label = label[:max_sent_length]
                label_id = label_id[:max_sent_length]
                cut_num += 1
            gazs = []
            gaz_ids = []
            w_length = len(char_id)
            for index in range(w_length):
                matched_list = gaz.enumerateMatchList(sent[index:])
                matched_length = [len(a) for a in matched_list]
                gazs.append(matched_list)
                matched_id = [gaz_alphabet.get_index(entity) for entity in matched_list]
                if matched_id:
                    gaz_ids.append([matched_id, matched_length])
                else:
                    gaz_ids.append([])
            instance_texts.append([sent, gazs, label])
            instance_ids.append([char_id, gaz_ids, label_id])
    return instance_ids, instance_texts, cut_num


def build_pretrain_embedding(embedding_path, alphabet, skip_first_row=False, separator=" ", embedd_dim=100, norm=True):
    embedd_dict = dict()
    if embedding_path != None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path, skip_first_row, separator)
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for alph, index in alphabet.iteritems():
        if alph in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[alph])
            else:
                pretrain_emb[index, :] = embedd_dict[alph]
            perfect_match += 1
        elif alph.lower() in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[alph.lower()])
            else:
                pretrain_emb[index, :] = embedd_dict[alph.lower()]
            case_match += 1
        else:
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    print("Embedding: %s\n     pretrain num:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s" % (
    embedding_path, pretrained_size, perfect_match, case_match, not_match, (not_match + 0.) / alphabet.size()))
    return pretrain_emb, embedd_dim


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec / root_sum_square


def load_pretrain_emb(embedding_path, skip_first_row=False, separator=" "):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r') as file:
        i = 0
        j = 0
        for line in file:
            if i == 0:
                i = i + 1
                if skip_first_row:
                    _ = line.strip()
                    continue
            j = j+1
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split(separator)
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                if embedd_dim + 1 == len(tokens):
                    embedd = np.empty([1, embedd_dim])
                    embedd[:] = tokens[1:]
                    embedd_dict[tokens[0]] = embedd
                else:
                    continue
    return embedd_dict, embedd_dim
