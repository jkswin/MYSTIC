"""
Adapted from https://github.com/DolbyUUU/byte_pair_encoding_BPE_subword_tokenization_implementation_python


- Removed dependence on BertTokenizer. 
- Simplified pre tokenizer that also uncases and removes punctuation except for ?!.
- Added save_pretrained()
- Added load_pretrained()
- Added tokenize_to_ids()
- Applies padding

"""

# install and import libraries
from collections import Counter, defaultdict
from string import ascii_lowercase, ascii_uppercase, whitespace
import json
import unicodedata
import os

ascii_characters_with_accents = ""

# Loop through all possible Unicode code points
for code_point in range(0x00C0, 0x0100):
    # Get the character for the code point
    character = chr(code_point)
    # Check if the character is a combining character or a letter with an accent
    if unicodedata.category(character) in ['Mn', 'Lm', 'Ll', 'Lu']:
        ascii_characters_with_accents += character

CHARS = ascii_lowercase + ascii_uppercase + whitespace + ascii_characters_with_accents + "0123456789?!"

class BPE():
    """Byte-Pair Encoding: Subword-based tokenization algorithm."""
    
    def __init__(self, corpus=None, vocab_size=None):
        """Initialize BPE tokenizer."""
        self.corpus = corpus
        self.vocab_size = vocab_size
        
        # pre-tokenize the corpus into words, BERT pre-tokenizer is used here
        self.word_freqs = defaultdict(int)
        self.vocab = []
        self.splits = {}
        self.merges = {}
        self.pad_token = "<PAD>"
        self.tok2id = {self.pad_token:0}
        self.id2tok = {0:self.pad_token}

    def tokenizer(self, sentence: list):
        """Pretokenize by whitespace. Also limit to roman alphabet + delim characters"""
        return "".join([char for char in sentence if char in CHARS]).lower().split()
    
    
    def train(self):
        """Train BPE tokenizer."""

        # compute the frequencies of each word in the corpus
        for text in self.corpus:
            for word in self.tokenizer(text):
                self.word_freqs[word] += 1

        # compute the base vocabulary of all characters in the corpus
        alphabet = []
        for word in self.word_freqs.keys():
            for letter in word:
                if letter not in alphabet:
                    alphabet.append(letter)
        alphabet.sort()

        # add the special token </w> at the beginning of the vocabulary
        vocab = ["</w>"] + alphabet.copy()

        # split each word into individual characters before training
        self.splits = {word: [c for c in word] for word in self.word_freqs.keys()}

        # merge the most frequent pair iteratively until the vocabulary size is reached
        while len(vocab) < self.vocab_size:

            # compute the frequency of each pair
            pair_freqs = self.compute_pair_freqs()

            # find the most frequent pair
            best_pair = ""
            max_freq = None
            for pair, freq in pair_freqs.items():
                if max_freq is None or max_freq < freq:
                    best_pair = pair
                    max_freq = freq

            # merge the most frequent pair
            self.splits = self.merge_pair(*best_pair)
            self.merges[best_pair] = best_pair[0] + best_pair[1]
            vocab.append(best_pair[0] + best_pair[1])

        self.vocab = vocab
        self.id2tok.update({i+1:tok for i, tok in enumerate(vocab)})
        self.tok2id.update({tok:i for i, tok in self.id2tok.items()})
        return self.merges


    def compute_pair_freqs(self):
        """Compute the frequency of each pair."""

        pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = self.splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs


    def merge_pair(self, a, b):
        """Merge the given pair."""

        for word in self.word_freqs:
            split = self.splits[word]
            if len(split) == 1:
                continue
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2 :]
                else:
                    i += 1
            self.splits[word] = split
        return self.splits
    

    def tokenize(self, text, max_len=100):
        """Tokenize a given text with trained BPE tokenizer (including pre-tokenization, split, and merge)."""
        
        pre_tokenized_text = self.tokenizer(text)
        splits_text = [[l for l in word] for word in pre_tokenized_text]

        for pair, merge in self.merges.items():
            for idx, split in enumerate(splits_text):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2 :]
                    else:
                        i += 1
                splits_text[idx] = split
        result = sum(splits_text, [])
        if len(result) > max_len:
            result = result[:max_len]
        elif len(result) < max_len:
            result += [self.pad_token] * (max_len - len(result))
            
        return result 
    
    def tokenize_to_ids(self, text, max_len=100):
        return [self.tok2id[token] for token in self.tokenize(text, max_len=max_len)]
    
    def save_pretrained(self, path="projects/text_adventure/models/bpe_model.json"):
        """Save a trained BPE model to a json file."""
        #convert self.merges to json friendly format
        json_merges = {v:k for k,v in self.merges.items()}

        with open(path, "w") as f:
            json.dump({
                "vocab": self.vocab,
                "id2tok": self.id2tok,
                "tok2id": self.tok2id,
                "merges": json_merges,
                "splits": self.splits,
                "pad_token": self.pad_token,
            }, f)

    @classmethod
    def from_pretrained(cls, path="projects/text_adventure/models/bpe_model.json"):
        """Load a saved BPE model from a JSON file"""
        bpe = cls()
        with open(path, 'r') as f:
            configs = json.load(f)
            for k,v in configs.items():
                if k == "merges":
                   v = {tuple(val):key for key,val in v.items()}
                setattr(bpe, k, v)
        return bpe
 

if __name__ == "__main__":

    root = "projects/text_adventure/"
    
    # load the pokemon and yugioh texts
    with open(root + "data/pokemon_text.json", "r", encoding="utf-8") as f, open(root + "data/yugioh_text.json", "r", encoding="utf-8") as f2:
        corpus1, corpus2 = json.load(f), json.load(f2)
        corpus = list(corpus1.values()) + list(corpus2.values())

    # load the spongebob transcripts
    sponge_dir = root + "data/spongebob_transcripts/"
    for file in os.listdir(sponge_dir):
        with open(sponge_dir + file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                if line and not line.startswith("This article is a transcript"):
                    # remove character indicators
                    corpus.append(line.partition(":")[-1])
    
    # load the book reviews, subtitles and other dialogue
    with open(root+"data/training_data.json", "r") as f:
        [corpus.append(t["review"]) for t in json.load(f)]

    print(len("".join(corpus)))

    bpe = BPE(corpus=corpus, vocab_size=5000)
    bpe.train()
    bpe.save_pretrained()

    loaded_bpe = BPE.from_pretrained()

    # tokenize the given text
    text = "I love Pokemon so so much. It's my favourite!"
    print(loaded_bpe.tokenize(text))
    print(loaded_bpe.tokenize_to_ids(text))