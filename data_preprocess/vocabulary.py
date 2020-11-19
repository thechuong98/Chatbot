# from preprocessing import EOS_TOKEN, PAD_TOKEN, SOS_TOKEN

PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2

class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_TOKEN: "PAD", SOS_TOKEN: "SOS", EOS_TOKEN: "EOS"}
        self.num_words = 3

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        if self.trimmed:
            return

        keep_words = []
        for word, word_count in self.word2count.items():
            if word_count >= min_count:
                keep_words.append(word)

        print('keep {}/{} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words)/len(self.word2index)
        ))   
        
        self.trimmed = True
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_TOKEN: "PAD",
                           SOS_TOKEN: "SOS", EOS_TOKEN: "EOS"}
        self.num_words = 3

        for word in keep_words:
            self.add_word(word)
