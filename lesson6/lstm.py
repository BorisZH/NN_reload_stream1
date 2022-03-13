import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy

from lesson6.data import DatasetSeq2Seq, collate_fn_seq2seq

data_dir = '/Users/a14419009/Repos/nn_reload_stream1/lesson6/dataset_text.txt'

dataset = DatasetSeq2Seq(data_dir)

#hyper params
input_vocab_len = len(dataset.input_sequnces_vocab)
output_vocab_len = len(dataset.output_sequnces_vocab)

cuda_device = -1
batch_size = 32
device = f'cuda:{cuda_device}' if cuda_device != -1 else 'cpu'


class Encoder(nn.Module):
    def __init__(self, vocab_len: int, emb_size: int = 256, hidden_size: int = 256):
        super().__init__()
        self.char_emb = nn.Embedding(vocab_len, emb_size)
        self.char_gru = nn.GRU(input_size=emb_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, x):
        embed = self.char_emb(x)
        out, hidden = self.char_gru(embed)

        return out, hidden


class Decoder(nn.Module):
    def __init__(self,
                 vocab_len: int,
                 eos_id: int,
                 emb_size: int = 256,
                 hidden_size: int = 256,

                 ):
        super().__init__()
        self.char_emb = nn.Embedding(vocab_len, emb_size)
        self.gru = nn.GRU(input_size=emb_size, hidden_size=hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, vocab_len)
        self.eos_id = eos_id

    def forward(self, hidden, sequence):
        if self.training:
            embed = self.char_emb(sequence)
            out, _ = self.gru(embed, hidden)

            return self.classifier(out)
        else:
            output_seq = []
            embed = self.char_emb(sequence)
            n_iter = 0
            char = -1
            while char != self.eos_id and n_iter < 20:
                out, hidden = self.gru(embed, hidden)
                char_proba = self.classifier(out)
                char = char_proba.argmax(-1)
                embed = self.char_emb(char)
                n_iter += 1
                output_seq.append(char.squeeze().item())

            return output_seq


class DateNormalizer(nn.Module):
    def __init__(self, input_voacb_len, output_vocab_len, emb_size, hidden_size, eos_id):
        super().__init__()
        self.encoder = Encoder(input_voacb_len, emb_size, hidden_size)
        self.decoder = Decoder(output_vocab_len, eos_id, emb_size, hidden_size)

    def forward(self, input_seq, output_seq):
        _, hidden = self.encoder(input_seq)
        decoded = self.decoder(hidden, output_seq)

        return decoded


model = DateNormalizer(input_vocab_len, output_vocab_len, 256, 256, dataset.output_sequnces_vocab['#']).to(device)
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
        collate_fn=collate_fn_seq2seq,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    for step, batch in enumerate(dataloader):
        input_seq = batch['data'].to(device)
        target = batch['target'].to(device)
        optim.zero_grad()
        predict = model(input_seq, target[:, :-1])

        loss = loss_func(predict.reshape(-1, output_vocab_len), target[:, 1:].reshape(-1))
        loss.backward()
        optim.step()
        if (step % 5 == 0):
            test = torch.tensor([[1, 19, 16, 27, 21, 29, 21, 22, 30, 27, 31, 32, 4, 18, 18, 4, 5, 6, 7, 4, 8, 3, 3, 28, 4, 23, 11,
                      2]])
            bos_tensor = torch.tensor([[dataset.output_sequnces_vocab['~']]])
            result = [1, 3, 4, 4, 8, 6, 7, 8, 6, 10, 10, 9, 2]
            with torch.no_grad():
                model.eval()
                test_predict = model(test, bos_tensor)
                model.train()
            print(loss, test_predict)

    print('Epoch {} finished'.format(epoch))