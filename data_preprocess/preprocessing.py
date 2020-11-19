import torch
from torch._C import StringType
from torch.nn.functional import normalize
from torch.utils.data import Dataset
import re #regular expression
import os
import codecs #encode and decode
import csv #deal with csv writer and reader 
import unicodedata
import itertools

from torchvision import transforms #deal with unicode strings
from vocabulary import Vocabulary
import sys

MAX_LENGTH = 10
MIN_COUNT = 3
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2

#read lines from movie_lines
def load_lines(filename, fields):
    """ return dict of lines with lineid is key and line object is value

    Args:
        filename (string): file name
        fields (list of strings): line objects fields

    Returns:
        lines: dict of lines with line_id is key and line_object is value
    """
    lines = {}
    with open(filename, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(' +++$+++ ')
            line_object = {}
            for i, field in enumerate(fields):
                line_object[field] = values[i]
            lines[line_object['lineID']] = line_object

    return lines


def load_conversation(filename, lines, fields):
    """
    input:
        file : u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L194', 'L195', 'L196', 'L197']
    returns:
        conversations : [{lines: [L194_object, L195_object, L196_object, L197_object]}]
    """
    conversations = []
    with open(filename, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(' +++$+++ ')
            conv_object = {}
            for i, field in enumerate(fields):
                conv_object[field] = values[i]
            utterance_id_pattern = re.compile('L[0-9]+')
            lines_ids = utterance_id_pattern.findall(conv_object['utteranceIDs'])
            conv_object['lines'] = []
            for line_idx in lines_ids:
                conv_object['lines'].append(lines[line_idx])
            conversations.append(conv_object)

    return conversations

def get_sentence_pairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        for i in range(len(conversation['lines'])-1):
            input_line = conversation['lines'][i]['text'].strip()
            target_line = conversation['lines'][i+1]['text'].strip()
            if input_line and target_line:
                qa_pairs.append([input_line, target_line])
    return qa_pairs

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s)
    return s

def read_vocabulary(datafile, corpus_name):
    print("Reading lines ...\n")
    lines = open(datafile, encoding='utf-8').read().strip().split('\n')
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]
    voc = Vocabulary(corpus_name)
    return voc, pairs

def pair_filter(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

def filter_pairs(pairs):
    return [pair for pair in pairs if pair_filter(pair)]

def load_data(corpus, corpus_name, data_file, save_dir):
    print("Preparing training data...")
    voc, pairs = read_vocabulary(data_file, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filter_pairs(pairs)
    print("Trimmed the pairs to {!s} pairs".format(len(pairs)))
    for pair in pairs:
        voc.add_sentence(pair[0])
        voc.add_sentence(pair[1])
    print("Counted words: ", voc.num_words)
    return voc, pairs

def trim_rare_word(voc: Vocabulary, pairs, MIN_COUNT):
    voc.trim(MIN_COUNT)
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs

    
def preprocessing():
    corpus_name = "cornell"
    corpus = os.path.join(corpus_name)
    datafile = os.path.join(sys.path[0], corpus, "formatted_movie_lines.txt")
    delimiter = '\t'
    delimiter = str(codecs.decode(delimiter, encoding='unicode_escape'))

    lines = {}
    conversations = {}

    MOVIE_LINES_FIELDS = ['lineID', 'characterID', 'movieID', 'character', 'text']
    MOVIE_CONVERSATIONS_FIELDS = ['character1ID', 'character2ID', 'movieID', 'utteranceIDs']

    print('\nProcessing the corpus ....')
    lines = load_lines(os.path.join(sys.path[0], corpus, 'movie_lines.txt'), MOVIE_LINES_FIELDS)
    print('\nLoading conversations...')
    conversations = load_conversation(os.path.join(sys.path[0], corpus, 'movie_conversations.txt'), lines, MOVIE_CONVERSATIONS_FIELDS)
    
    print("\nWriting formatted file ...")
    with open(datafile, 'w', encoding='utf-8') as f:
        #write to csv file
        writer = csv.writer(f, delimiter=delimiter, lineterminator='\n')
        for pair in get_sentence_pairs(conversations):
            writer.writerow(pair)

    save_dir = os.path.join("corpus", "save")
    voc, pairs = load_data(corpus, corpus_name, datafile, save_dir)
    pairs = trim_rare_word(voc, pairs, MIN_COUNT=3)

    ###TO-DO : convert pairs into tensors with a maximum length

    
    ####

    return voc, pairs

def indexes_from_sentence(voc: Vocabulary, sentence):
    """Convert sentence to list of indexes

    Args:
        voc ([type]): [description]
        sentence ([type]): [description]
    """
    return [voc.word2index(word) for word in sentence.split(' ')] + [EOS_TOKEN] 




class CornellDataset(Dataset):
    def __init__(self, voc, pairs, transform = None):
        self.voc = voc
        self.pairs = pairs
        self.transform = transform
    
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        
