import math, random, time, torch, re
from pathlib import Path
from torch.autograd import Variable
from .Models import get_model
from .Process import *
import torch.nn.functional as F
from .Optim import CosineWithRestarts
from .Batch import create_masks, nopeak_mask
import dill as pickle


class Trainer:
    def __init__(self, src_train_data_path, trg_train_data_path, src_valid_data_path, trg_valid_data_path, src_lang,
                 trg_lang, max_strlen, batchsize, load_weights, checkpoint, device, train, valid, src_pad, trg_pad, SRC,
                 TRG, train_len, valid_len):
        self.src_train_data_path = src_train_data_path
        self.trg_train_data_path = trg_train_data_path
        self.src_valid_data_path = src_valid_data_path
        self.trg_valid_data_path = trg_valid_data_path
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.max_strlen = max_strlen
        self.batchsize = batchsize
        self.load_weights = load_weights
        self.checkpoint = checkpoint
        self.device = device
        self.train = train
        self.valid = valid
        self.src_pad = src_pad
        self.trg_pad = trg_pad
        self.SRC = SRC
        self.TRG = TRG
        self.train_len = train_len
        self.valid_len = valid_len

    def init_model(self, d_model, lr, heads, dropout, n_layers, load_weights, SGDR, device):
        self.d_model = d_model
        self.lr = lr
        self.heads = heads
        self.dropout = dropout
        self.n_layers = n_layers
        self.load_weights = load_weights
        self.SGDR = SGDR
        self.device = device
        self.model = get_model(len(self.SRC.vocab), len(self.TRG.vocab), self.d_model, self.heads, self.dropout,
                               self.n_layers, self.load_weights, self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-9)
        if self.SGDR == True:
            self.sched = CosineWithRestarts(self.optimizer, T_max=self.train_len)

    def train_model(self, epochs, printevery, model_save_dir):

        self.printevery = printevery
        self.model_save_dir = Path(model_save_dir)

        if self.checkpoint > 0:
            print(
                f"model weights will be saved every {self.checkpoint} minutes and at end of epoch to directory weights/")

        if not os.path.isdir(self.model_save_dir):
            os.mkdir(self.model_save_dir)

        pickle.dump(self.SRC, open(f'{self.model_save_dir}/SRC.pkl', 'wb'))
        pickle.dump(self.TRG, open(f'{self.model_save_dir}/TRG.pkl', 'wb'))

        print("training model...")
        start = time.time()
        if self.checkpoint > 0:
            cptime = time.time()

        for epoch in range(epochs):
            total_loss = 0
            total_loss_valid = 0

            if self.checkpoint > 0:
                torch.save(self.model.state_dict(), f'{self.model_save_dir}/model_weights')

            for i, batch in enumerate(self.train):
                self.model.train()
                total_loss += self.fit_batch(batch)
                avg_loss = total_loss / (i + 1)

                if self.checkpoint > 0 and ((time.time() - cptime) // 60) // self.checkpoint >= 1:
                    torch.save(self.model.state_dict(), f'{self.model_save_dir}/model_weights')
                    cptime = time.time()

            self.model.eval()
            for j, batch in enumerate(self.valid):
                total_loss_valid += self.fit_batch(batch, train=False)
                avg_loss_valid = total_loss_valid / (j + 1)

            print("%dm: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, loss = %.03f, loss valid = %.03f" % \
                  ((time.time() - start) // 60, epoch + 1, "".join('#' * (100 // 5)), "".join(' ' * (20 - (100 // 5))),
                   100, avg_loss, epoch + 1, avg_loss, avg_loss_valid))

        print(f"saving weights to {self.model_save_dir}/...")
        torch.save(self.model.state_dict(), f'{self.model_save_dir}/model_weights')

    def fit_batch(self, batch, train=True):

        src = batch.src.transpose(0, 1)
        trg = batch.trg.transpose(0, 1)
        trg_input = trg[:, :-1]
        src_mask, trg_mask = create_masks(src, trg_input, self.src_pad, self.trg_pad, self.device)
        preds = self.model(src, trg_input, src_mask, trg_mask)
        ys = trg[:, 1:].contiguous().view(-1)
        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=self.trg_pad)
        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.SGDR == True and train:
            self.sched.step()

        return loss.item()

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)

    def init_vars(self, src, k=3, max_len=80):

        init_tok = self.TRG.vocab.stoi['<sos>']
        src = src.to(self.device)
        src_mask = (src != self.SRC.vocab.stoi['<pad>']).unsqueeze(-2)
        e_output = self.model.encoder(src, src_mask)

        outputs = torch.Tensor([[init_tok]]).long()
        outputs = outputs.to(device=self.device)
        trg_mask = nopeak_mask(1, self.device)
        out = self.model.out(self.model.decoder(outputs, e_output, src_mask, trg_mask))
        out = F.softmax(out, dim=-1)
        probs, ix = out[:, -1].data.topk(k)
        log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)
        outputs = torch.zeros(k, max_len).long()
        outputs = outputs.to(device=self.device)
        outputs[:, 0] = init_tok
        outputs[:, 1] = ix[0]
        e_outputs = torch.zeros(k, e_output.size(-2), e_output.size(-1))
        e_outputs = e_outputs.to(self.device)
        e_outputs[:, :] = e_output[0]

        return outputs, e_outputs, log_scores

    def beam_search(self, src, k=3, max_len=80):
        src = src.to(self.device)
        outputs, e_outputs, log_scores = self.init_vars(src, k, max_len)
        eos_tok = self.TRG.vocab.stoi['<eos>']
        src_mask = (src != self.SRC.vocab.stoi['<pad>']).unsqueeze(-2).to(self.device)
        ind = None
        for i in range(2, max_len):
            trg_mask = nopeak_mask(i, self.device)
            out = self.model.out(self.model.decoder(outputs[:, :i], e_outputs, src_mask, trg_mask))
            out = F.softmax(out, dim=-1)
            outputs, log_scores = self.k_best_outputs(outputs, out, log_scores, i, k)

            if (outputs == eos_tok).nonzero().size(0) == k:
                alpha = 0.7
                div = 1 / ((outputs == eos_tok).nonzero()[:, 1].type_as(log_scores) ** alpha)
                _, ind = torch.max(log_scores * div, 1)
                ind = ind.data[0]
                break

        if ind is None:
            ind = 0
        lengths = (outputs[ind] == eos_tok).nonzero()

        if len(lengths) == 0:  # there isn't any eos token, take till padding starts
            lengths = reversed((outputs[ind]).nonzero())
        length = lengths[0]
        return ' '.join([self.TRG.vocab.itos[tok] for tok in outputs[ind][1:length]])

    def k_best_outputs(self, outputs, out, log_scores, i, k):

        probs, ix = out[:, -1].data.topk(k)
        log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0, 1)
        k_probs, k_ix = log_probs.view(-1).topk(k)
        row = k_ix // k
        col = k_ix % k
        outputs[:, :i] = outputs[row, :i]
        outputs[:, i] = ix[row, col]

        log_scores = k_probs.unsqueeze(0)
        return outputs, log_scores

    def fix_punctuation(self, text):
        text = re.sub(r'\s([?.,!"](?:\s|$))', r'\1', text)
        text = re.sub(' +', ' ', text)
        return text

    def predict(self, sentence, fix_punctuation=True):
        self.model.eval()
        indexed = []
        sentence = self.SRC.preprocess(sentence)
        for tok in sentence:
            print(f'token {tok}')
            try:
                if self.SRC.vocab.stoi[tok] != 0:
                    print(f' token i {self.SRC.vocab.stoi[tok]}')
                    indexed.append(self.SRC.vocab.stoi[tok])
            # this is quick fix some reason is crashes in flask app context if finds token not in vocab, instead of returning 0
            except Exception as e:
                indexed.append(0)
        sentence = Variable(torch.LongTensor([indexed]))
        sentence = self.beam_search(sentence)
        if fix_punctuation:
            sentence = self.fix_punctuation(sentence)
        return sentence

    def get_param_dict(self):
        dct = {'src_train_data_path': self.src_train_data_path,
               'trg_train_data_path': self.trg_train_data_path,
               'src_valid_data_path': self.src_valid_data_path,
               'trg_valid_data_path': self.trg_valid_data_path,
               'src_lang': self.src_lang,
               'trg_lang': self.trg_lang,
               'max_strlen': self.max_strlen,
               'batchsize': self.batchsize,
               'load_weights': self.load_weights,
               'checkpoint': self.checkpoint,
               'device': self.device,
               'src_pad': self.src_pad,
               'trg_pad': self.trg_pad,
               'train_len': self.train_len,
               'valid_len': self.valid_len,
               'd_model': self.d_model,
               'lr': self.lr,
               'heads': self.heads,
               'dropout': self.dropout,
               'n_layers': self.n_layers,
               'SGDR': self.SGDR}
        return dct

    def save_model(self):
        self.model = None
        self.train = None
        self.valid = None
        self.SRC = None
        self.TRG = None
        params = self.get_param_dict()
        torch.save(params, self.model_save_dir / 'model_params.pth')

    @staticmethod
    def load_model(folder, device='cpu'):
        folder = Path(folder)
        params = torch.load(folder / 'model_params.pth')
        model = Trainer(params.get('src_train_data_path'), params.get('trg_train_data_path'),
                        params.get('src_valid_data_path'),
                        params.get('trg_valid_data_path'), params.get('src_lang'),
                        params.get('trg_lang'), params.get('max_strlen'), params.get('batchsize'),
                        params.get('load_weights'),
                        params.get('checkpoint'), device, params.get('train'), params.get('valid'),
                        params.get('src_pad'), params.get('trg_pad'), params.get('SRC'),
                        params.get('TRG'), params.get('train_len'), params.get('valid_len'))
        model.SRC = pickle.load(open(folder / 'SRC.pkl', 'rb'))
        model.TRG = pickle.load(open(folder / 'TRG.pkl', 'rb'))

        model.init_model(params.get('d_model'), params.get('lr'), params.get('heads'),
                         params.get('dropout'), params.get('n_layers'), params.get('load_weights'),
                         params.get('SGDR'), device)

        model.model.load_state_dict(torch.load(folder / 'model_weights', map_location=device))
        return model

    @classmethod
    def create_from_txt(cls, src_train_data_path, trg_train_data_path, src_valid_data_path, trg_valid_data_path,
                        src_lang, trg_lang, max_strlen, batchsize, load_weights, checkpoint, device):
        src_train_data, trg_train_data = read_data(src_train_data_path, trg_train_data_path)
        src_valid_data, trg_valid_data = read_data(src_valid_data_path, trg_valid_data_path)
        SRC, TRG = create_fields(src_lang, trg_lang, load_weights)
        train, valid, src_pad, trg_pad, train_len, valid_len = create_dataset(src_train_data, trg_train_data,
                                                                              src_valid_data,
                                                                              trg_valid_data, SRC, TRG, max_strlen,
                                                                              batchsize, device)

        return cls(src_train_data_path, trg_train_data_path, src_valid_data_path, trg_valid_data_path, src_lang,
                   trg_lang, max_strlen, batchsize, load_weights, checkpoint, device, train, valid, src_pad,
                   trg_pad, SRC, TRG, train_len, valid_len)
