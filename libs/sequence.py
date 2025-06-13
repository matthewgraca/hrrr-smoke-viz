# NOTE this should be accompanied by changes in the preprocessing pipeline
# for instance, each channel should be processed, saved, then freed.

# I can also imagine a future in which each channel might become to large to process in RAM...
# we'll get there when we get there. But similarly, we'd process the data in batches (then 
# do a second pass since scaling requires statistics from the whole channel.
# e.g. process + create batches, collecting statistics. Then a second pass over each batch to scale.

from keras.utils import PyDataset
import numpy as np
import math

class PWWBPyDataset(PyDataset):
    def __init__(self, X_paths, y_path, batch_size, **kwargs):
        '''
        The sequence reads the numpy files (each file is a channel) 
        lazily; when it comes time to train using the batch, it is then
        evaluated.

        The data flows from disk to RAM to GPU VRAM in batches.

        Usage:
        # preprocess and save np files...

        # initialize generator
        channel_paths = ['c1.npy', 'c2.npy', ... ] 
        label_path = ['label.npy']
        batch_size = 4
        generator = PWWBPyDataset(
            channel_paths, label_path, batch_size,  
            workers=8, use_multiprocessing=True
        )
        model.fit(generator, ...) 
        '''
        super().__init__(**kwargs)
        # lazy load the np arrays
        self.channels = [np.load(path, mmap_mode='r') for path in X_paths]
        self.labels = np.load(y_path, mmap_mode='r')
        self.batch_size = batch_size

        for channel in self.channels:
            assert len(channel) == len(self.labels), \
                "All channels and labels need the same number of samples."

    def __len__(self):
        '''
        Number of batches
        '''
        return math.ceil(len(self.labels) / self.batch_size)

    def __getitem__(self, index):
        '''
        Each item is a completely loaded batch.

        e.g. 
        - channel 1 and channel 2, from indices 1:32, are loaded and stacked
        - the label for this batch is also returned
        '''
        # last batch may be smaller if total items aren't cleanly divisible by batch size
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.labels))

        # load each channel
        channel_batch = [channel[start:end] for channel in self.channels]
        x = np.stack(channel_batch, axis=-1)
        y = self.labels[start:end]

        return x, y
