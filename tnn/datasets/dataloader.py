import os
from torch.utils.data.dataloader import DataLoader


class sDataLoader(DataLoader):

    def get_stream(self):
        while True:
            for data in sDataLoader.__iter__(self):
                yield data
