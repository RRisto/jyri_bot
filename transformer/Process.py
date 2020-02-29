import os
import pandas as pd
from torchtext import data
from .Tokenize import tokenize
from .Batch import MyIterator, batch_size_fn
import dill as pickle


def read_data(src_data_path, trg_data_path):
    if src_data_path is not None:
        try:
            src_data = open(src_data_path).read().strip().split('\n')
        except:
            print("error: '" + src_data + "' file not found")
            quit()

    if trg_data_path is not None:
        try:
            trg_data = open(trg_data_path).read().strip().split('\n')
        except:
            print("error: '" + trg_data + "' file not found")
            quit()
    return src_data, trg_data


def create_fields(src_lang, trg_lang, load_weights=None):
    spacy_langs = ['en', 'fr', 'de', 'es', 'pt', 'it', 'nl']
    if src_lang not in spacy_langs:
        print('invalid src language: ' + src_lang + 'supported languages : ' + spacy_langs)
    if trg_lang not in spacy_langs:
        print('invalid trg language: ' + trg_lang + 'supported languages : ' + spacy_langs)

    print("loading spacy tokenizers...")

    t_src = tokenize(src_lang)
    t_trg = tokenize(trg_lang)

    TRG = data.Field(lower=True, tokenize=t_trg.tokenizer, init_token='<sos>', eos_token='<eos>')
    SRC = data.Field(lower=True, tokenize=t_src.tokenizer)

    if load_weights is not None:
        try:
            print("loading presaved fields...")
            SRC = pickle.load(open(f'{load_weights}/SRC.pkl', 'rb'))
            TRG = pickle.load(open(f'{load_weights}/TRG.pkl', 'rb'))
        except:
            print("error opening SRC.pkl and TRG.pkl field files, please ensure they are in " + load_weights + "/")
            quit()

    return (SRC, TRG)


def create_dataset(src_train_data, trg_train_data, src_valid_data, trg_valid_data, SRC, TRG, max_strlen, batchsize,
                   device):
    print("creating dataset and iterator... ")

    raw_train_data = {'src': [line for line in src_train_data], 'trg': [line for line in trg_train_data]}
    df_train = pd.DataFrame(raw_train_data, columns=["src", "trg"])
    mask_train = (df_train['src'].str.count(' ') < max_strlen) & (df_train['trg'].str.count(' ') < max_strlen)
    df_train = df_train.loc[mask_train]
    df_train.to_csv("translate_transformer_train_temp.csv", index=False)

    raw_valid_data = {'src': [line for line in src_valid_data], 'trg': [line for line in trg_valid_data]}
    df_valid = pd.DataFrame(raw_valid_data, columns=["src", "trg"])
    mask_valid = (df_valid['src'].str.count(' ') < max_strlen) & (df_valid['trg'].str.count(' ') < max_strlen)
    df_valid = df_valid.loc[mask_valid]
    df_valid.to_csv("translate_transformer_valid_temp.csv", index=False)

    data_fields = [('src', SRC), ('trg', TRG)]
    # train = data.TabularDataset('./translate_transformer_temp.csv', format='csv', fields=data_fields)

    train, valid = data.TabularDataset.splits(path='', train='translate_transformer_train_temp.csv',
                                              validation='translate_transformer_valid_temp.csv', format='csv',
                                              fields=data_fields)

    train_iter = MyIterator(train, batch_size=batchsize, device=device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True, shuffle=True)

    valid_iter = MyIterator(valid, batch_size=batchsize, device=device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False, shuffle=True)

    os.remove('translate_transformer_train_temp.csv')
    os.remove('translate_transformer_valid_temp.csv')

    SRC.build_vocab(train)
    TRG.build_vocab(train)

    src_pad = SRC.vocab.stoi['<pad>']
    trg_pad = TRG.vocab.stoi['<pad>']

    train_len = get_len(train_iter)
    valid_len = get_len(valid_iter)

    return train_iter, valid_iter, src_pad, trg_pad, train_len, valid_len


def get_len(train):
    for i, b in enumerate(train):
        pass

    return i
