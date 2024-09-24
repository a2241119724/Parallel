import json
import torch

class Vocab():
    def __init__(self) -> None:
        self.i2vocab = json.load(open("./data/vocab/i2vocab.json", "r", encoding="utf-8"))
        self.vocab2i = json.load(open("./data/vocab/vocab2i.json", "r", encoding="utf-8"))
        self.vocab_size = len(self.i2vocab.keys())

    def vocab2i_fn(self, vocab):
        if vocab in self.vocab2i.keys():
            return self.vocab2i[vocab]
        else:
            return self.vocab2i["<unk>"]

    def i2caption(self, caption2i):
        caption2i = caption2i.tolist()
        i2vocab = []
        for cs in caption2i:
            _t = []
            for c in cs:
                if c == self.vocab2i["<eos>"]:
                    break
                _t.append(self.i2vocab[str(c)])
            i2vocab.append(" ".join(_t))
        return i2vocab
    
    def caption2i(self, caption:str):
        caption = caption.split(" ")
        vocab2i = []
        for i in range(len(caption)):
            if caption[i] in self.vocab2i:
                vocab2i.append(self.vocab2i[caption[i]])
            else: # 遇到词表中没有的词
                vocab2i.append(self.vocab2i["<unk>"])
                # print(f"添加{caption[i]}到词表中")
                # self.vocab2i[caption[i]] = self.vocab_size
                # self.i2vocab[str(self.vocab_size)] = caption[i]
                # self.vocab_size = self.vocab_size + 1
        return vocab2i
    
    def vocab_size(self):
        return self.vocab_size
    
    def __len__(self):
        return self.vocab_size
    
