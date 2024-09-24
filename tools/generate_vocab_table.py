import json
import sys
import os

sys.path.append(os.path.abspath("./"))

from models.utils import pre_handle_caption

def _handle_caption_info(vocab2count, threshold:int,sentence_lengths:dict):
    print('vocab total: ', len(vocab2count.keys()))
    print('vocab total(repeat): ', sum(vocab2count.values()))
    # 小于阈值的单词
    unk_vocabs = [v for v,n in vocab2count.items() if n <= threshold]
    unk_vocabs_count = sum(vocab2count[w] for w in unk_vocabs)
    print('number of unk vocabs: %.2f%%' % (len(unk_vocabs)*100./len(vocab2count)))
    print('number of unk vocabs(repeat): %.2f%%' % (unk_vocabs_count*100./sum(vocab2count.values())))
    print("-------------------")
    max_len = int(max(sentence_lengths.keys()))
    print('max length sentence: ', max_len)
    sum_len = sum(sentence_lengths.values())
    for i in range(max_len+1):
        if(sentence_lengths.get(i,0) != 0):
            print('长度为%2d的句子占比: %f%%' % (i, sentence_lengths.get(i,0)*100.0/sum_len))

'''
RuntimeError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
'''
def _generate_vocab_table(vocab:list):
    print('know vocab total: ', len(vocab))
    print("start generate vocab table...")
    vocab2i = {}
    i2vocab = {}
    vocab2i["<pad>"] = 0
    i2vocab[0] = "<pad>"
    vocab2i["<unk>"] = 1
    i2vocab[1] = "<unk>"
    vocab2i["<bos>"] = 2
    i2vocab[2] = "<bos>"
    vocab2i["<eos>"] = 3
    i2vocab[3] = "<eos>"
    dec:int = 4
    for i in range(len(vocab)):
        if vocab[i] == "":
            dec = dec - 1
            continue
        vocab2i[vocab[i]] = i + dec
        i2vocab[i + dec] = vocab[i]
    # save
    json.dump(vocab2i, open("./data/vocab/vocab2i.json", "w", encoding="utf-8"))
    json.dump(i2vocab, open("./data/vocab/i2vocab.json", "w", encoding="utf-8"))
    print("end generate vocab table...")
    
def _generate_sentence_max_len(sentence_lengths:dict,len_threshold:float):
    print("start generate sentence max len...")
    max_len = int(max(sentence_lengths.keys()))
    sum_len = sum(sentence_lengths.values())
    rotio = 0.
    for i in range(max_len+1):
        rotio = rotio + sentence_lengths.get(i,0)/sum_len
        if rotio >= len_threshold:
            print("句子最大长度阈值为: %d" % (i))
            break
    print("end generate sentence max len...")

'''
* @name: generate_vocab_table.py
* @description: 生成词表
* @param {type threshold:int} 词频阈值
* @return {type}
'''
def handle_caption(count_threshold:int,len_threshold:float):
    train_anno = json.load(open("./data/annotations/coco_lab_karpathy_train2014.json", "r", encoding="utf-8"))
    val_anno = json.load(open("./data/annotations/coco_lab_karpathy_val2014.json", "r", encoding="utf-8"))
    test_anno = json.load(open("./data/annotations/coco_lab_karpathy_test2014.json", "r", encoding="utf-8"))
    # 每个单词出现的次数
    vocab2count = {}
    # 每个句子的长度
    sentence_lengths = {}
    for anno in train_anno:
        caption = anno["caption"].split(" ")
        sentence_lengths[len(caption)] = sentence_lengths.get(len(caption), 0) + 1
        for c in caption:
            vocab2count[c] = vocab2count.get(c, 0) + 1
    # for anno in val_anno:
    #     for caption in anno["caption"]:
    #         caption = caption.split(" ")
    #         sentence_lengths[len(caption)] = sentence_lengths.get(len(caption), 0) + 1
    #         for c in caption:
    #             vocab2count[c] = vocab2count.get(c, 0) + 1
    # for anno in test_anno:
    #     for caption in anno["caption"]:
    #         caption = caption.split(" ")
    #         sentence_lengths[len(caption)] = sentence_lengths.get(len(caption), 0) + 1
    #         for c in caption:
    #             vocab2count[c] = vocab2count.get(c, 0) + 1
    
    _handle_caption_info(vocab2count,count_threshold,sentence_lengths)
    # 排序
    count2vocab = sorted([(count,vocab) for vocab,count in vocab2count.items()], reverse=True)
    k_vocabs = [v for n,v in count2vocab if n > count_threshold]
    _generate_vocab_table(k_vocabs)
    _generate_sentence_max_len(sentence_lengths,len_threshold)

if __name__ == "__main__":
    pre_handle_caption()
    handle_caption(5,0.9999)
    # handle_caption(10,0.999)