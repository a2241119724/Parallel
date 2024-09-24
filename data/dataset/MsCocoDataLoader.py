from torch.utils.data import DataLoader
from models.enums import DataType
from config.GlobalConfig import GlobalConfig

class MsCocoDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, sampler=None, drop_last=False, num_workers=0):
        if dataset.data_type in { DataType.TEST, DataType.VAL }:
            super(MsCocoDataLoader, self).__init__(dataset, batch_size, shuffle, 
                drop_last=drop_last, num_workers=num_workers, collate_fn=dataset.collate_val_test_fn())
        elif dataset.data_type == DataType.TRAIN:
            super(MsCocoDataLoader, self).__init__(dataset, batch_size, shuffle, 
                drop_last=drop_last, sampler=sampler, num_workers=num_workers, collate_fn=dataset.collate_sg_train_fn())