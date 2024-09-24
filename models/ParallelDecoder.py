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

class StackDecoderOne(nn.Module):
    def __init__(self, d_model:int, dropout) -> None:
        super().__init__()
        self.lstm1 = nn.LSTMCell(d_model + d_model, d_model)
        # self.lstm1 = nn.LSTM(d_model, d_model, 1, batch_first=True)
        self.attention = Attention(d_model)
        self.lstm2 = nn.LSTMCell(d_model + d_model, d_model)
        # self.lstm2 = nn.LSTM(d_model, d_model, 1, batch_first=True)

    def init(self):
        # nn.init.xavier_uniform_(self.lstm1.weight_ih)
        # nn.init.xavier_uniform_(self.lstm1.weight_hh)
        # nn.init.xavier_uniform_(self.lstm2.weight_ih)
        # nn.init.xavier_uniform_(self.lstm2.weight_hh)

        # nn.init.constant_(self.lstm1.bias_ih, 0.)
        # nn.init.constant_(self.lstm1.bias_hh, 0.)
        # nn.init.constant_(self.lstm2.bias_ih, 0.)
        # nn.init.constant_(self.lstm2.bias_hh, 0.)

        self.attention.init()

    def forward(self, next_word, enc_output_mean, enc_output, state, sg_mask):
        enc_hid = (enc_output_mean + state[0][1]) / 2.
        next_word = torch.cat((enc_hid, next_word), dim=1)
        h_1, c_1 = self.lstm1(next_word, (state[0][0], state[1][0]))
        # 
        v = self.attention(enc_output, h_1, mask=sg_mask)
        #
        h_1_new = torch.cat((h_1, v), dim=1)
        h_2, c_2 = self.lstm2(h_1_new, (state[0][1], state[1][1]))
        #
        state = (torch.stack([h_1, h_2]), torch.stack([c_1, c_2]))
        return h_2, state

class StackDecoderTwo(nn.Module):
    def __init__(self, d_model:int, dropout, num_heads) -> None:
        super().__init__()
        self.lstm1 = nn.LSTMCell(d_model + d_model, d_model)
        # self.lstm1 = nn.LSTM(d_model, d_model, 1, batch_first=True)
        self.attention = Attention(d_model)
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

    def init(self):
        nn.init.xavier_uniform_(self.linear[0].weight)
        nn.init.xavier_uniform_(self.linear1[0].weight)
        nn.init.xavier_uniform_(self.linear2[0].weight)
        nn.init.xavier_uniform_(self.linear3[0].weight)
        # nn.init.xavier_uniform_(self.lstm1.weight_ih)
        # nn.init.xavier_uniform_(self.lstm1.weight_hh)

        nn.init.constant_(self.linear[0].bias, 0.)
        nn.init.constant_(self.linear1[0].bias, 0.)
        nn.init.constant_(self.linear2[0].bias, 0.)
        nn.init.constant_(self.linear3[0].bias, 0.)
        # nn.init.constant_(self.lstm1.bias_ih, 0.)
        # nn.init.constant_(self.lstm1.bias_hh, 0.)

        self.attention.init()
        self.multi_head_attention.init()

    def forward(self, next_word, enc_output_mean, enc_output, state, sg_mask):
        enc_hid = (enc_output_mean + state[0][1]) / 2.
        next_word = torch.cat((enc_hid, next_word), dim=1)
        h_3, c_3 = self.lstm1(next_word, (state[0][0], state[1]))
        # 
        v = self.attention(enc_output, h_3, mask=sg_mask)
        # AoA
        fv = self.multi_head_attention(h_3.unsqueeze(1), enc_output, sg_mask) + h_3.unsqueeze(1)
        fv = torch.cat((fv, v.unsqueeze(1)), dim=-1)
        i = self.linear(fv) + self.linear2(h_3.unsqueeze(1))
        g = F.sigmoid(self.linear1(fv) + self.linear3(h_3.unsqueeze(1)))
        h_3_new = g * i
        h_3_new = h_3_new.squeeze(1)
        state = (torch.stack([h_3, h_3_new]), c_3)
        return h_3_new, state

class ParallelDecoderLayer(nn.Module):
    def __init__(self, d_model: int, N_dec:int, dropout, num_heads) -> None:
        super().__init__()
        self.stack_decoder_one_layers = nn.ModuleList([StackDecoderOne(d_model, dropout) for _ in range(N_dec)])
        self.stack_decoder_two_layers = nn.ModuleList([StackDecoderTwo(d_model, dropout, num_heads) for _ in range(N_dec)])

    def init(self):
        for l in self.stack_decoder_one_layers:
            l.init()
        for l in self.stack_decoder_two_layers:
            l.init()

    def forward(self, next_word, enc_output_mean, enc_output, state, sg_mask):
        x_1, x_2 = None, None
        state_1 = (state[0][:2], state[1][:2])
        state_2 = (state[0][2:], state[1][2])
        for l in self.stack_decoder_one_layers:
            x_1, state_1 = l(next_word, enc_output_mean, enc_output, state_1, sg_mask)
        for l in self.stack_decoder_two_layers:
            x_2, state_2 = l(next_word, enc_output_mean, enc_output, state_2, sg_mask)
        state = (torch.cat([state_1[0], state_2[0]]), torch.cat([state_1[1], state_2[1].unsqueeze(0)]))
        next_word = torch.cat((x_1, x_2), dim=-1)
        # next_word = x_1 * x_2
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

    def init(self):
        nn.init.xavier_uniform_(self.linear[0].weight)
        nn.init.xavier_uniform_(self.init_h[0].weight)
        nn.init.xavier_uniform_(self.init_c[0].weight)

        # nn.init.constant_(self.linear[0].bias, 0.)
        nn.init.constant_(self.init_h[0].bias, 0.)
        nn.init.constant_(self.init_c[0].bias, 0.)

        for l in self.layers:
            l.init()

    ## 每次不softmax
    def forward(self, 
            enc_output_mean:torch.Tensor, 
            enc_output:torch.Tensor, 
            caption2i:torch.Tensor, 
            caption2vector:torch.Tensor,
            sg_mask:torch.Tensor,
            obj_obj:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """_summary_
        Args:
            enc_output_mean (torch.Tensor): _description_
            enc_output (torch.Tensor): _description_
            caption2i (torch.Tensor): 第一个位置为<cls>
            caption2vector (torch.Tensor): _description_
            sg_mask (torch.Tensor): _description_
        Returns:
            _type_: _description_
        """
        # 将<pad>的位置遮盖为0
        # mask_pos = torch.BoolTensor([[False] * GlobalConfig.max_seq_len]).to(enc_output_mean.device)
        # pos = torch.arange(1, GlobalConfig.max_seq_len + 1).view(1, -1).expand(self.batch_size, -1).to(enc_output_mean.device)
        # mask_pos = (caption2i == GlobalConfig.padding_idx)
        # pos = pos.masked_fill(mask_pos, 0)
        # pos2vector = self.pos_emb(pos)
        # output
        output = torch.ones((enc_output.shape[0], 1, GlobalConfig.vocab_size)).to(enc_output_mean.device) * GlobalConfig.token_bos
        out_caption2i = torch.zeros((enc_output.shape[0], 1), dtype=torch.long).to(enc_output.device)
        # 保存lstm隐藏状态
        state = self._init_hidden_state(enc_output_mean)
        for i in range(GlobalConfig.max_seq_len - 1):
            caption2i_one = caption2vector[:,i,:].clone().contiguous()
            for l in self.layers:
                caption2i_one, state = l(caption2i_one, enc_output_mean, enc_output, state, sg_mask)
            caption2i_one = self.linear(caption2i_one)
            # caption2i_one = caption2i_one * self.temperature
            output = torch.cat([output, caption2i_one.unsqueeze(1)], dim=1)
            # caption2i_one = self.softmax(caption2i_one)
            caption2i_one = F.log_softmax(caption2i_one, -1)
            # caption2i_one = torch.multinomial(caption2i_one, 1)
            caption2i_one = torch.argmax(caption2i_one, dim=1)
            out_caption2i = torch.cat([out_caption2i, caption2i_one.unsqueeze(-1).long()], dim=-1)
        return output, out_caption2i
    
    def sample(self, enc_output_mean, enc_output, sg_mask, beam_size=3, obj_obj=None):
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
                    caption2i_one = self.word_emb(last_word)
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
        return (h.unsqueeze(0).repeat(4, 1, 1), c.unsqueeze(0).repeat(3, 1, 1))
