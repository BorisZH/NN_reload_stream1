import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# with open('/Users/a14419009/Repos/nn_reload_stream1/lesson6/dataset_text.txt', 'r') as f:


class DatasetSeq2Seq(Dataset):
    def __init__(self, file_name, train_lang='en', bos: str = '~', eos: str = '#'):

        with open(file_name, 'r') as f:
            train = f.readlines()

        self.input_sequnces_vocab = {'pad': 0, bos: 1, eos: 2}
        self.output_sequnces_vocab = {'pad': 0, bos: 1, eos: 2}

        self.input_sequnces = []
        self.output_sequnces = []

        n_input = 3
        n_output = 3
        for line in train:
            split_line = line.split('\t')

            sequence = [self.input_sequnces_vocab[bos]]

            for char in split_line[0]:
                if self.input_sequnces_vocab.get(char) is None:
                    self.input_sequnces_vocab[char] = n_input
                    n_input += 1
                sequence.append(self.input_sequnces_vocab[char])
            sequence.append(self.input_sequnces_vocab[eos])

            target = [self.output_sequnces_vocab[bos]]
            for char in split_line[2]:
                if self.output_sequnces_vocab.get(char) is None:
                    self.output_sequnces_vocab[char] = n_output
                    n_output += 1
                target.append(self.output_sequnces_vocab[char])
            target.append(self.output_sequnces_vocab[eos])

            self.input_sequnces.append(sequence)
            self.output_sequnces.append(target)

        self.target_decode = [k for k in self.output_sequnces_vocab.keys()]

    def __len__(self):
        return len(self.input_sequnces)

    def __getitem__(self, index):
        return {
            'data': self.input_sequnces[index],
            'target': self.output_sequnces[index],
        }


def collate_fn_seq2seq(input_data):
    data = []
    targets = []

    for item in input_data:
        data.append(torch.as_tensor(item['data']))
        targets.append(torch.as_tensor(item['target']))

    data = pad_sequence(data, batch_first=True, padding_value=0)
    targets = pad_sequence(targets, batch_first=True, padding_value=0)
    data_mask = data > 0
    targets_mask = targets > 0

    return {'data': data, 'target': targets, 'data_mask': data_mask, 'targets_mask': targets_mask}


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    data_dir = '/Users/a14419009/Repos/nn_reload_stream1/lesson6/dataset_text.txt'

    dataset = DatasetSeq2Seq(data_dir)

    dataloader = DataLoader(
            dataset=dataset,
            collate_fn=collate_fn_seq2seq,
            batch_size=4,
            shuffle=True,
            drop_last=True,
        )
    for step, batch in enumerate(dataloader):
        if step == 0:
            print(batch)
        else:
            break