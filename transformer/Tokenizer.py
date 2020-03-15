import re
import sentencepiece as spm

st = spm.SentencePieceProcessor()
st.Load('tokenizers/bpe_vocab_size_10000_norm_identity.model')

class Tokenizer(object):

    def __init__(self, tokenizer_path):
        pass
        #self.st=st
        #self.st.Load(tokenizer_path)

    def tokenizer(self, sentence):
        sentence = re.sub(
            r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower()
        return st.EncodeAsPieces(sentence)
