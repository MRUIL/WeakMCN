# coding=utf-8

import torch
import torch.nn as nn

from models.network_blocks import SA,AttFlat
from utils.utils import make_mask


class LSTM_SA(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size):
        super(LSTM_SA, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.GRU(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
            dropout=__C.DROPOUT_R,
            bidirectional=False
        )

        self.sa_list = nn.ModuleList([SA(__C) for _ in range(__C.N_SA)])
        self.att_flat=AttFlat(__C)
        if __C.EMBED_FREEZE:
            self.frozen(self.embedding)
    def frozen(self, module):
        if getattr(module, 'module', False):
            for child in module.module():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = False
    def forward(self, ques_ix):

        # Pre-process Language Feature
        lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)


        for sa in self.sa_list:
            lang_feat = sa(lang_feat, lang_feat_mask)

        flat_lang_feat = self.att_flat(lang_feat, lang_feat_mask)
        return  {
            'flat_lang_feat':flat_lang_feat,
            'lang_feat':lang_feat,
            'lang_feat_mask':lang_feat_mask
        }


backbone_dict={
    'lstm':LSTM_SA,
}

def language_encoder(__C, pretrained_emb, token_size):
    lang_enc=backbone_dict[__C.LANG_ENC](__C, pretrained_emb, token_size)
    return lang_enc