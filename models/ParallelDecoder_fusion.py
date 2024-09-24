import torch
import spacy
import torch.nn.functional as F

from models.enums import DataType
from torch import nn
from config.GlobalConfig import GlobalConfig
from .model import MultiHeadAttention, Attention, MultiHeadAttention_new, MeshedMemoryMultiHeadAttention_new
from .utils import get_activation_function
from .utils import sinusoid_encoding_table
from models.utils import CalcTime
from collections import namedtuple
from data.vocab.Vocab import Vocab
from typing import Tuple

BeamCandidate = namedtuple('BeamCandidate',
                           ['state', 'log_prob_sum', 'log_prob_seq', 'last_word_id', 'word_id_seq'])

class ParallelDecoderLayer(nn.Module):
    def __init__(self, d_model: int, N_dec:int, dropout, num_heads) -> None:
        super().__init__()
        self.lstm1 = nn.LSTMCell(d_model + d_model, d_model)
        # self.lstm1 = nn.LSTM(d_model, d_model, 1, batch_first=True)
        self.attention = Attention(d_model)
        self.lstm2 = nn.LSTMCell(d_model + d_model, d_model)
        # self.lstm2 = nn.LSTM(d_model, d_model, 1, batch_first=True)
        # self.multi_head_attention = MultiHeadAttention(d_model, 64, num_heads, True, dropout)
        self.multi_head_attention = MeshedMemoryMultiHeadAttention_new(d_model, 64, num_heads, False, True, dropout)
        self.linear = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            get_activation_function("relu"),
            nn.Dropout(p=GlobalConfig.dropout),
        )
        self.linear1 = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            get_activation_function("relu"),
            nn.Dropout(p=GlobalConfig.dropout),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(d_model, d_model),
            get_activation_function(),
            nn.Dropout(p=GlobalConfig.dropout),
        )
        self.linear3 = nn.Sequential(
            nn.Linear(d_model, d_model),
            get_activation_function(),
            # nn.Dropout(p=GlobalConfig.dropout),
        )
        self.param = nn.Parameter(torch.zeros(d_model, 100))

    def init(self):
        # nn.init.xavier_uniform_(self.lstm1.weight_hh)
        # nn.init.xavier_uniform_(self.lstm2.weight_ih)
        # nn.init.xavier_uniform_(self.lstm2.weight_hh)

        # nn.init.constant_(self.lstm1.bias_ih, 0.)
        # nn.init.constant_(self.lstm1.bias_hh, 0.)
        # nn.init.constant_(self.lstm2.bias_ih, 0.)
        # nn.init.constant_(self.lstm2.bias_hh, 0.)

        nn.init.xavier_uniform_(self.param)
        nn.init.xavier_uniform_(self.linear[0].weight)
        nn.init.xavier_uniform_(self.linear1[0].weight)
        nn.init.xavier_uniform_(self.linear2[0].weight)
        nn.init.xavier_uniform_(self.linear3[0].weight)

        nn.init.constant_(self.linear[0].bias, 0.)
        nn.init.constant_(self.linear1[0].bias, 0.)
        nn.init.constant_(self.linear2[0].bias, 0.)
        nn.init.constant_(self.linear3[0].bias, 0.)

        self.attention.init()
        self.multi_head_attention.init()

    def forward(self, next_word, enc_output_mean, enc_output, state, sg_mask):
        enc_hid = (enc_output_mean + state[0][1] + state[0][2]) / 3.
        next_word = torch.cat((enc_hid, next_word), dim=1)
        h_1, c_1 = self.lstm1(next_word, (state[0][0], state[1][0]))
        #
        h_1_new = F.softmax(torch.matmul(h_1, self.param), dim=-1)
        h_1_new = torch.matmul(h_1_new, self.param.transpose(-1, -2))
        h_1_new = F.dropout(F.relu(h_1_new)) + h_1
        #
        v, _q = self.attention(enc_output, h_1_new, mask=sg_mask)
        # layer1
        h_1_new_new = torch.cat((h_1_new, v), dim=1)
        h_2, c_2 = self.lstm2(h_1_new_new, (state[0][1], state[1][1]))
        # layer2, AoA
        fv = self.multi_head_attention(h_1_new.unsqueeze(1), enc_output, sg_mask) + h_1_new.unsqueeze(1)
        fv = torch.cat((fv, v.unsqueeze(1)), dim=-1)
        i = self.linear(fv) + self.linear2(_q.unsqueeze(1))
        g = F.sigmoid(self.linear1(fv) + self.linear3(_q.unsqueeze(1)))
        h_3 = (g * i).squeeze(1)
        state = (torch.stack([h_1, h_2, h_3]), torch.stack([c_1, c_2]))
        next_word = torch.cat((h_2, h_3), dim=-1)
        return next_word, state

class ParallelDecoder(nn.Module):
    def __init__(self, d_model: int,  N_dec:int, word_emb, dropout, num_heads) -> None:
        super().__init__()
        self.d_model = d_model
        self.word_emb = word_emb
        self.vocab = Vocab()
        # self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(GlobalConfig.max_seq_len + 1, d_model, 0), freeze=True)
        self.layers = nn.ModuleList([ParallelDecoderLayer(GlobalConfig.d_model, N_dec, dropout, num_heads) for _ in range(1)])
        self.linear = nn.Sequential(
            nn.Linear(d_model * 2, GlobalConfig.vocab_size, bias=False),
            get_activation_function(),
            # nn.Dropout(p=GlobalConfig.dropout),
        )
        self.temperature = 1.2
        self.init_h = nn.Sequential(
            nn.Linear(d_model, d_model),
            get_activation_function(),
            nn.Dropout(p=GlobalConfig.dropout),
        )
        self.init_c = nn.Sequential(
            nn.Linear(d_model, d_model),
            get_activation_function(),
            nn.Dropout(p=GlobalConfig.dropout),
        )
        self.multi_head_attention = MeshedMemoryMultiHeadAttention_new(d_model, 64, num_heads, False, True, dropout)

    def init(self):
        nn.init.xavier_uniform_(self.linear[0].weight)
        nn.init.xavier_uniform_(self.init_h[0].weight)
        nn.init.xavier_uniform_(self.init_c[0].weight)

        # nn.init.constant_(self.linear[0].bias, 0.)
        nn.init.constant_(self.init_h[0].bias, 0.)
        nn.init.constant_(self.init_c[0].bias, 0.)

        for l in self.layers:
            l.init()
        self.multi_head_attention.init()

    ## 每次不softmax
    def forward(self, 
            enc_output_mean:torch.Tensor, 
            enc_output:torch.Tensor, 
            caption2i:torch.Tensor, 
            caption2vector:torch.Tensor,
            sg_mask:torch.Tensor, sg, _sg_mask) -> Tuple[torch.Tensor, torch.Tensor]:
        # 将<pad>的位置遮盖为0
        # mask_pos = torch.BoolTensor([[False] * GlobalConfig.max_seq_len]).to(enc_output_mean.device)
        # pos = torch.arange(1, GlobalConfig.max_seq_len + 1).view(1, -1).expand(self.batch_size, -1).to(enc_output_mean.device)
        # mask_pos = (caption2i == GlobalConfig.padding_idx)
        # pos = pos.masked_fill(mask_pos, 0)
        # pos2vector = self.pos_emb(pos)
        # output
        output = torch.ones((enc_output.shape[0], 1, GlobalConfig.vocab_size)).to(enc_output_mean.device) * GlobalConfig.token_bos
        # 保存lstm隐藏状态
        state = self._init_hidden_state(enc_output_mean)
        for i in range(GlobalConfig.max_seq_len - 1):
            caption2i_one = caption2vector[:,i,:].clone().contiguous().unsqueeze(1)
            caption2i_one = F.layer_norm(self.multi_head_attention(caption2i_one, sg, _sg_mask) + caption2i_one, (caption2i_one.shape[-1],)).squeeze(1)
            for l in self.layers:
                caption2i_one, state = l(caption2i_one, enc_output_mean, enc_output, state, sg_mask)
            # caption2i_one = caption2i_one * self.temperature
            output = torch.cat([output, caption2i_one.unsqueeze(1)], dim=1)
            # caption2i_one = torch.multinomial(caption2i_one, 1)
        # caption2i_one = self.softmax(caption2i_one)
        output = self.linear(output)
        return F.log_softmax(output, -1), torch.argmax(output, dim=1)
    
    def sample(self, enc_output_mean, enc_output, sg_mask, beam_size=3, sg=None, _sg_mask=None):
        # 将<pad>的位置遮盖为0
        # mask_pos = torch.BoolTensor([[False] * GlobalConfig.max_seq_len]).to(self.device)
        # pos = torch.arange(1, GlobalConfig.max_seq_len + 1).view(1, -1).expand(self.batch_size, -1).to(self.device)
        # pos = pos.masked_fill(mask_pos, 0)
        # pos2vector = self.pos_emb(pos)
        # 保存lstm隐藏状态
        state = self._init_hidden_state(enc_output_mean)
        # state, log_prob_sum, log_prob_seq, last_word_id, word_id_seq
        candidates = [BeamCandidate(state, 0., [], GlobalConfig.token_bos, [])]
        for i in range(GlobalConfig.max_seq_len - 1):
            tmp_candidates = []
            end_flag = True
            for candidate in candidates:
                state, log_prob_sum, log_prob_seq, last_word_id, word_id_seq = candidate
                if i > 0 and last_word_id == GlobalConfig.token_eos:
                    tmp_candidates.append(candidate)
                else:
                    end_flag = False
                    last_word = torch.tensor([last_word_id], dtype=torch.long).to(enc_output_mean.device)
                    caption2i_one = self.word_emb(last_word).unsqueeze(1)
                    caption2i_one = F.layer_norm(self.multi_head_attention(caption2i_one, sg, _sg_mask) + caption2i_one, (caption2i_one.shape[-1],)).squeeze(1)
                    for l in self.layers:
                        caption2i_one, state = l(caption2i_one, enc_output_mean, enc_output, state, sg_mask)
                    caption2i_one = self.linear(caption2i_one)
                    # caption2i_one = self.softmax(caption2i_one)
                    caption2i_one = F.log_softmax(caption2i_one, -1)
                    logprobs = caption2i_one.squeeze(0)
                    ## do not generate <PAD>, <SOS> and <UNK>
                    logprobs[GlobalConfig.token_bos] = float('-inf')
                    # logprobs[GlobalConfig.token_end] = float('-inf')
                    logprobs[GlobalConfig.padding_idx] = float('-inf')
                    ## do not generate last step word
                    logprobs[last_word_id] = float('-inf')
                    output_sorted, index_sorted = torch.sort(logprobs, descending=True)
                    for k in range(beam_size):
                        log_prob, word_id = output_sorted[k], index_sorted[k]
                        log_prob = float(log_prob)
                        word_id = int(word_id)
                        tmp_candidates.append(BeamCandidate(state, log_prob_sum + log_prob,
                                                            log_prob_seq + [log_prob],
                                                            word_id, word_id_seq + [word_id]))
            candidates = sorted(tmp_candidates, key=lambda x: x.log_prob_sum, reverse=True)[:beam_size]
            if end_flag:
                break
         # captions, scores
        captions = [' '.join([self.vocab.i2vocab[str(idx)] for idx in candidate.word_id_seq if idx != GlobalConfig.token_eos])
                    for candidate in candidates]
        scores = [candidate.log_prob_sum for candidate in candidates]
        return captions, scores
    
    def _init_hidden_state(self, enc_output_mean):
        h = self.init_h(enc_output_mean)
        c = self.init_c(enc_output_mean)
        return (h.unsqueeze(0).repeat(3, 1, 1), c.unsqueeze(0).repeat(2, 1, 1))
