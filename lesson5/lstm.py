import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy

from lesson5.data import DatasetSeq, collate_fn

data_dir = '/raid/home/bgzhestkov/nn_reload/lesson4/'
train_lang = 'en'


dataset = DatasetSeq(data_dir)

#hyper params
vocab_len = len(dataset.word_vocab) + 1
n_classes = len(dataset.target_vocab) + 1
n_chars = len(dataset.char_vocab) + 1
cuda_device = 14
batch_size = 4
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


class POSPredictor(nn.Module):
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
        chars_out = []
        for char in chars:
            # B x Tc x C
            tmp = self.char_rnn(char.to(x.device)).squeeze().unsqueeze(1)
            chars_out.append(tmp)
        # B x T x C_emb
        chars_out = torch.cat(chars_out, dim=1)
        embedded = self.word_emb(x)
        out, _ = self.gru(torch.cat([embedded, chars_out], dim=-1))

        return self.classifier(out)


model = POSPredictor(vocab_len, n_classes, n_chars).to(device)
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
        data = batch['data'].to(device)
        optim.zero_grad()
        predict = model(data, batch['chars'])

        loss = loss_func(predict.view(-1, n_classes), batch['target'].to(device).view(-1))
        loss.backward()
        optim.step()
        if (step % 50 == 0):
            print(loss)

