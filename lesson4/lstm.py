import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy



data_dir = '/Users/a14419009/Repos/nn_reload_stream1/lesson4/'
train_lang = 'en'


class DatasetSeq(Dataset):
    def __init__(self, data_dir, train_lang='en'):

        with open(data_dir + train_lang + '.train', 'r') as f:
            train = f.read().split('\n\n')

        # delete extra tag markup
        train = [x for x in train if not '_ ' in x]

        self.target_vocab = {}
        self.word_vocab = {}
        self.char_vocab = {}

        self.encoded_sequences = []
        self.encoded_targets = []
        self.encoded_char_sequences = []
        n_word = 0
        n_target = 0
        n_char = 0
        for line in train:
            sequence = []
            target = []
            chars = []
            for item in line.split('\n'):
                if item != '':
                    word, label = item.split(' ')

                    if self.word_vocab.get(word) is None:
                        self.word_vocab[word] = n_word
                        n_word += 1
                    if self.target_vocab.get(label) is None:
                        self.target_vocab[label] = n_target
                        n_target += 1
                    for char in word:
                        if self.char_vocab.get(char) is None:
                            self.char_vocab[char] = n_char
                            n_char += 1
                    sequence.append(self.word_vocab[word])
                    target.append(self.target_vocab[label])
                    chars.append([self.char_vocab[char] for char in word])
            self.encoded_sequences.append(sequence)
            self.encoded_targets.append(target)
            self.encoded_char_sequences.append(chars)

    def __len__(self):
        return len(self.encoded_sequences)

    def __getitem__(self, index):
        return {
            'data': torch.tensor(self.encoded_sequences[index]),
            'char': torch.tensor(self.encoded_char_sequences[index]),
            'target': torch.tensor(self.encoded_targets[index]),
        }

dataset = DatasetSeq(data_dir)

#hyper params
vocab_len = len(dataset.word_vocab)
n_classes = len(dataset.target_vocab)
n_chars = len(dataset.char_vocab)
cuda_device = -1
batch_size = 1
device = f'cuda:{cuda_device}' if cuda_device != -1 else 'cpu'


#model
class CharModel(nn.Module):
    def __init__(self, char_vocab_len: int, emb_size: int = 128, hidden_size: int = 128):
        super().__init__()
        self.char_emb = nn.Embedding(char_vocab_len, emb_size)
        self.char_gru = nn.GRU(input_size=emb_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, x):
        embed = self.char_emb(x)
        _, out = self.char_gru(embed)

        return out


class POS_predictor(nn.Module):
    def __init__(self,
                 word_vocab_len: int,
                 n_classes: int,
                 char_vocab_len: int,
                 emb_size: int = 128,
                 hidden_size: int = 128,
                 char_emb_size: int = 64,
                 char_hidden_size: int = 64,
                 ):
        super().__init__()
        self.word_emb = nn.Embedding(word_vocab_len, emb_size)
        self.char_rnn = CharModel(char_vocab_len=char_vocab_len, emb_size=char_emb_size, hidden_size=char_hidden_size)

        self.gru = nn.GRU(input_size=emb_size+char_hidden_size, hidden_size=hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, x, chars):
        n_words = x.size(1)
        chars_out = []
        for id_ in range(n_words):
            # B x T x C
            input_t = chars[:, id_, :].squeeze()
            #B x 1 x C_emb
            chars_out.append(self.char_rnn(input_t).unsqueeze(1))
        # B x T x C_emb
        chars_out = torch.cat(chars_out, dim=1)
        embedded = self.word_emb(x)
        out, _ = self.gru(torch.cat([embedded, chars_out], dim=-1))

        return self.classifier(out)


def collate_fn(data):

    return data[0]


model = POS_predictor(vocab_len, n_classes, n_chars)
#optimizer
optim = torch.optim.Adam(model.parameters(), lr=0.001)
#lr scheduler


#
# import matplotlib.pyplot as plt
# plt.imshow(dataset.data[0].detach().numpy())
# plt.show()
#loss
loss_func = nn.CrossEntropyLoss()
#dataloder
for epoch in range(20):
    dataloader = DataLoader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    for step, batch in enumerate(dataloader):
        data = batch['data'].to(device).unsqueeze(0)
        optim.zero_grad()
        predict = model(data)

        loss = loss_func(predict.view(-1, n_classes), batch['target'].to(device))
        loss.backward()
        optim.step()
        if (step % 50 == 0):
            print(loss)

