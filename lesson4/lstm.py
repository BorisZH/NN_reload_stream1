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

        self.encoded_sequences = []
        self.encoded_targets = []
        n_word = 0
        n_target = 0
        for line in train:
            sequence = []
            target = []
            for item in line.split('\n'):
                if item != '':
                    word, label = item.split(' ')

                    if self.word_vocab.get(word) is None:
                        self.word_vocab[word] = n_word
                        n_word += 1
                    if self.target_vocab.get(label) is None:
                        self.target_vocab[label] = n_target
                        n_target += 1

                    sequence.append(self.word_vocab[word])
                    target.append(self.target_vocab[label])
            self.encoded_sequences.append(sequence)
            self.encoded_targets.append(target)

    def __len__(self):
        return len(self.encoded_sequences)

    def __getitem__(self, index):
        return {
            'data': torch.tensor(self.encoded_sequences[index]),
            'target': torch.tensor(self.encoded_targets[index]),
        }

dataset = DatasetSeq(data_dir)

#hyper params
vocab_len = len(dataset.word_vocab)
n_classes = len(dataset.target_vocab)
cuda_device = -1
batch_size = 1
device = f'cuda:{cuda_device}' if cuda_device != -1 else 'cpu'


#model
class POS_predictor(nn.Module):
    def __init__(self, word_vocab_len: int, n_classes: int, emb_size: int = 128, hidden_size: int = 128):
        super().__init__()
        self.word_emb = nn.Embedding(emb_size, word_vocab_len)
        self.gru = nn.GRU(input_size=emb_size, hidden_size=hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        embedded = self.word_emb(x)
        out, _ = self.gru(embedded)

        return self.classifier(out)


def collate_fn(data):

    return data


model = POS_predictor(vocab_len, n_classes)
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

