import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from layer.crf import CRF
from layer.gatlayer import GAT
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class BLSTM_GAT_CRF(nn.Module):
    def __init__(self, data, args):
        super(BLSTM_GAT_CRF, self).__init__()
        print("build batched BLSTM_GAT_CRF...")
        self.name = "BLSTM_GAT_CRF"
        self.strategy = args.strategy
        self.char_emb_dim = data.char_emb_dim
        self.gaz_emb_dim = data.gaz_emb_dim
        self.gaz_embeddings = nn.Embedding(data.gaz_alphabet.size(), self.gaz_emb_dim)
        self.char_embeddings = nn.Embedding(data.char_alphabet.size(), self.char_emb_dim)
        if data.pretrain_char_embedding is not None:
            self.char_embeddings.weight.data.copy_(torch.from_numpy(data.pretrain_char_embedding))
        else:
            self.char_embeddings.weight.data.copy_(
                torch.from_numpy(self.random_embedding(data.char_alphabet.size(), self.char_emb_dim)))
        if data.pretrain_gaz_embedding is not None:
            self.gaz_embeddings.weight.data.copy_(torch.from_numpy(data.pretrain_gaz_embedding))
        else:
            self.gaz_embeddings.weight.data.copy_(
                torch.from_numpy(self.random_embedding(data.gaz_alphabet.size(), self.gaz_emb_dim)))
        if args.fix_gaz_emb:
            self.gaz_embeddings.weight.requires_grad = False
        else:
            self.gaz_embeddings.weight.requires_grad = True
        self.hidden_dim = self.gaz_emb_dim
        self.bilstm_flag = args.bilstm_flag
        self.lstm_layer = args.lstm_layer
        if self.bilstm_flag:
            lstm_hidden = self.hidden_dim // 2
        else:
            lstm_hidden = self.hidden_dim
        crf_input_dim = data.label_alphabet.size()+1
        self.lstm = nn.LSTM(self.char_emb_dim, lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)
        self.hidden2hidden = nn.Linear(self.hidden_dim, crf_input_dim)
        self.gat_1 = GAT(self.hidden_dim, args.gat_nhidden, crf_input_dim, args.dropgat, args.alpha, args.gat_nhead, args.gat_layer)
        self.gat_2 = GAT(self.hidden_dim, args.gat_nhidden, crf_input_dim, args.dropgat, args.alpha, args.gat_nhead, args.gat_layer)
        self.gat_3 = GAT(self.hidden_dim, args.gat_nhidden, crf_input_dim, args.dropgat, args.alpha, args.gat_nhead, args.gat_layer)
        self.crf = CRF(data.label_alphabet.size()-1, args.use_gpu)
        if self.strategy == "v":
            self.weight1 = nn.Parameter(torch.ones(crf_input_dim))
            self.weight2 = nn.Parameter(torch.ones(crf_input_dim))
            self.weight3 = nn.Parameter(torch.ones(crf_input_dim))
            self.weight4 = nn.Parameter(torch.ones(crf_input_dim))
        elif self.strategy == "n":
            self.weight1 = nn.Parameter(torch.ones(1))
            self.weight2 = nn.Parameter(torch.ones(1))
            self.weight3 = nn.Parameter(torch.ones(1))
            self.weight4 = nn.Parameter(torch.ones(1))
        else:
            self.weight = nn.Linear(crf_input_dim*4, crf_input_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.droplstm = nn.Dropout(args.droplstm)
        self.gaz_dropout = nn.Dropout(args.gaz_dropout)
        self.reset_parameters()
        if args.use_gpu:
            self.to_cuda()

    def to_cuda(self):
        self.char_embeddings = self.char_embeddings.cuda()
        self.gaz_embeddings = self.gaz_embeddings.cuda()
        self.lstm = self.lstm.cuda()
        self.gat_1 = self.gat_1.cuda()
        self.gat_2 = self.gat_2.cuda()
        self.gat_3 = self.gat_3.cuda()
        self.hidden2hidden = self.hidden2hidden.cuda()
        self.gaz_dropout = self.gaz_dropout.cuda()
        self.dropout = self.dropout.cuda()
        self.droplstm = self.droplstm.cuda()
        self.gaz_dropout = self.gaz_dropout.cuda()
        if self.strategy in ["v", "n"]:
            self.weight1.data = self.weight1.data.cuda()
            self.weight2.data = self.weight2.data.cuda()
            self.weight3.data = self.weight3.data.cuda()
            self.weight4.data = self.weight4.data.cuda()
        else:
            self.weight = self.weight.cuda()

    def reset_parameters(self):
        nn.init.orthogonal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0_reverse)
        nn.init.orthogonal_(self.lstm.weight_ih_l0_reverse)
        nn.init.orthogonal_(self.hidden2hidden.weight)
        nn.init.constant_(self.hidden2hidden.bias, 0)

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def _get_lstm_features(self, batch_char, batch_len):
        embeds = self.char_embeddings(batch_char)
        embeds = self.dropout(embeds)
        embeds_pack = pack_padded_sequence(embeds, batch_len, batch_first=True)
        out_packed, (_, _) = self.lstm(embeds_pack)
        lstm_feature, _ = pad_packed_sequence(out_packed, batch_first=True)
        lstm_feature = self.droplstm(lstm_feature)
        return lstm_feature

    def _get_crf_feature(self, batch_char, batch_len, gaz_list, t_graph, c_graph, l_graph):
        gaz_feature = self.gaz_embeddings(gaz_list)
        gaz_feature = self.gaz_dropout(gaz_feature)
        lstm_feature = self._get_lstm_features(batch_char, batch_len)
        max_seq_len = lstm_feature.size()[1]
        gat_input = torch.cat((lstm_feature, gaz_feature), dim=1)
        gat_feature_1 = self.gat_1(gat_input, t_graph)
        gat_feature_1 = gat_feature_1[:, :max_seq_len, :]
        gat_feature_2 = self.gat_2(gat_input, c_graph)
        gat_feature_2 = gat_feature_2[:, :max_seq_len, :]
        gat_feature_3 = self.gat_3(gat_input, l_graph)
        gat_feature_3 = gat_feature_3[:, :max_seq_len, :]
        lstm_feature = self.hidden2hidden(lstm_feature)
        if self.strategy == "m":
            crf_feature = torch.cat((lstm_feature, gat_feature_1, gat_feature_2, gat_feature_3), dim=2)
            crf_feature = self.weight(crf_feature)
        elif self.strategy == "v":
            crf_feature = torch.mul(lstm_feature, self.weight1) + torch.mul(gat_feature_1, self.weight2) + torch.mul(
                gat_feature_2, self.weight3) + torch.mul(gat_feature_3, self.weight4)
        else:
            crf_feature = self.weight1 * lstm_feature + self.weight2 * gat_feature_1 + self.weight3 * gat_feature_2 + self.weight4 * gat_feature_3
        return crf_feature

    def neg_log_likelihood(self, batch_char, batch_len, gaz_list, t_graph, c_graph, l_graph, mask, batch_label):
        crf_feature = self._get_crf_feature(batch_char, batch_len, gaz_list, t_graph, c_graph, l_graph)
        total_loss = self.crf.neg_log_likelihood_loss(crf_feature, mask, batch_label)
        return total_loss

    def forward(self, batch_char, batch_len, gaz_list, t_graph, c_graph, l_graph, mask):
        crf_feature = self._get_crf_feature(batch_char, batch_len, gaz_list, t_graph, c_graph, l_graph)
        _, best_path = self.crf._viterbi_decode(crf_feature, mask)
        return best_path

