from models.enums import DataType

class GlobalConfig():
    # text token
    padding_idx:int = 0
    token_unk:int = 1
    token_bos:int = 2
    token_eos:int = 3
    max_seq_len:int = 15 + 2
    # max_seq_len:int = 28 + 2
    vocab_size:int = 0
    # image
    d_model:int = 512
    #
    dropout:float = 0.1
    # 不同模式下代码不同
    mode:DataType = DataType.TRAIN