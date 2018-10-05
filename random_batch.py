"""
   filename                             chapter_id speaker_id dataset_id
0  1272/128104/1272-128104-0000.wav     128104       1272  dev-clean
1  1272/128104/1272-128104-0001.wav     128104       1272  dev-clean
2  1272/128104/1272-128104-0002.wav     128104       1272  dev-clean
3  1272/128104/1272-128104-0003.wav     128104       1272  dev-clean
4  1272/128104/1272-128104-0004.wav     128104       1272  dev-clean
5  1272/128104/1272-128104-0005.wav     128104       1272  dev-clean
6  1272/128104/1272-128104-0006.wav     128104       1272  dev-clean
7  1272/128104/1272-128104-0007.wav     128104       1272  dev-clean
8  1272/128104/1272-128104-0008.wav     128104       1272  dev-clean
9  1272/128104/1272-128104-0009.wav     128104       1272  dev-clean
"""

import numpy as np
import pandas as pd

import constants as c
from pre_process import data_catalog


def clipped_audio(x, num_frames=c.NUM_FRAMES):
    if x.shape[0] > num_frames:
        bias = np.random.randint(0, x.shape[0] - num_frames)
        clipped_x = x[bias: num_frames + bias]
    else:
        clipped_x = x

    return clipped_x


class MiniBatch:
    def __init__(self, libri, batch_size, unique_speakers=None):    #libri['filename']，libri['chapter_id']，libri['speaker_id']，libri['dataset_id']
        # indices = np.random.choice(len(libri), size=batch_size, replace=False)
        # [anc1, anc2, anc3, pos1, pos2, pos3, neg1, neg2, neg3]
        # [sp1, sp2, sp3, sp1, sp2, sp3, sp4, sp5, sp6]
        if unique_speakers is None:
            unique_speakers = list(libri['speaker_id'].unique())
        num_triplets = batch_size

        anchor_batch = None
        positive_batch = None
        negative_batch = None
        for ii in range(num_triplets):
            two_different_speakers = np.random.choice(unique_speakers, size=2, replace=False)
            anchor_positive_speaker = two_different_speakers[0]
            negative_speaker = two_different_speakers[1]
            anchor_positive_file = libri[libri['speaker_id'] == anchor_positive_speaker].sample(n=2, replace=False)
            anchor_df = pd.DataFrame(anchor_positive_file[0:1])
            anchor_df['training_type'] = 'anchor'
            positive_df = pd.DataFrame(anchor_positive_file[1:2])
            positive_df['training_type'] = 'positive'
            negative_df = libri[libri['speaker_id'] == negative_speaker].sample(n=1)
            negative_df['training_type'] = 'negative'

            if anchor_batch is None:
                anchor_batch = anchor_df.copy()
            else:
                anchor_batch = pd.concat([anchor_batch, anchor_df], axis=0)
            if positive_batch is None:
                positive_batch = positive_df.copy()
            else:
                positive_batch = pd.concat([positive_batch, positive_df], axis=0)
            if negative_batch is None:
                negative_batch = negative_df.copy()
            else:
                negative_batch = pd.concat([negative_batch, negative_df], axis=0)

        self.libri_batch = pd.DataFrame(pd.concat([anchor_batch, positive_batch, negative_batch], axis=0))
        self.num_triplets = num_triplets

    def to_inputs(self):

        new_x = []
        for i in range(len(self.libri_batch)):
            filename = self.libri_batch[i:i + 1]['filename'].values[0]
            x = np.load(filename)
            new_x.append(clipped_audio(x))
        x = np.array(new_x) #(batchsize, num_frames, 64, 1)
        y = self.libri_batch['speaker_id'].values

        # anchor examples [speakers] == positive examples [speakers]
        np.testing.assert_array_equal(y[0:self.num_triplets], y[self.num_triplets:2 * self.num_triplets])

        return x, y


def stochastic_mini_batch(libri, batch_size=c.BATCH_SIZE,unique_speakers=None):
    mini_batch = MiniBatch(libri, batch_size,unique_speakers)
    return mini_batch


def main():
    libri = data_catalog(c.DATASET_DIR)
    batch = stochastic_mini_batch(libri, c.BATCH_SIZE)

    x, y = batch.to_inputs()
    print(x.shape,y.shape)


if __name__ == '__main__':
    main()
