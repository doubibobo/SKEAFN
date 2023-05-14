import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from einops import rearrange


def l2norm(t, groups=1):
    t = rearrange(t, '... (g d) -> ... g d', g=groups)
    t = F.normalize(t, p=2, dim=-1)
    return rearrange(t, '... g d -> ... (g d)')


class SinusoidalPositionEmbedding(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class AbsolutePositionalEmbedding(nn.Module):

    def __init__(self, dim, max_seq_len, l2norm_embed=False):
        super().__init__()
        self.scale = dim**-0.5 if not l2norm_embed else 1.
        self.max_seq_len = max_seq_len
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos=None):
        seq_len = x.shape[1]
        assert seq_len <= self.max_seq_len, f'you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}'

        if pos is None:
            pos = torch.arange(seq_len, device=x.device)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return l2norm(pos_emb) if self.l2norm_embed else pos_emb


class ScaleShift(nn.Module):

    def __init__(self, time_emb_dim, dim_out):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.GELU(),
                                      nn.Linear(time_emb_dim, dim_out * 2))
        self.init_zero_(self.time_mlp[-1])

    def init_zero_(self, layer):
        nn.init.constant_(layer.weight, 0.)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.)

    def forward(self, x, time_emb):
        scale, shift = self.time_mlp(time_emb).chunk(2, dim=2)
        x = x * (scale + 1) + shift
        return x


class MeanPooling(nn.Module):
    """
    参考自: https://www.kaggle.com/code/quincyqiang/feedback-meanpoolingv2-inference
    """

    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones(last_hidden_state.shape[0:-1])
            attention_mask = attention_mask.to(last_hidden_state.device)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class AddNormalBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()

        # layernormal linear
        self.ln = nn.LayerNorm(dim)

    def forward(self, seq1, seq2):
        output_seq = seq1 + seq2
        output_seq = self.ln(output_seq)
        return output_seq


class FeedForward(nn.Module):

    def __init__(self, input_dim, inner_dim, outer_dim, drop_rate=0.2):
        super().__init__()

        # inner_linear
        self.fc1 = nn.Linear(input_dim, inner_dim)
        self.act = nn.GELU()

        # drop_layer
        self.drp_layer = nn.Dropout(p=drop_rate)

        # outer_linear
        self.fc2 = nn.Linear(inner_dim, outer_dim)

    def forward(self, input_seq):
        output_seq = self.act(self.fc1(input_seq))
        output_seq = self.drp_layer(output_seq)
        output_seq = self.fc2(output_seq)
        return output_seq


class EncoderLayer(nn.Module):

    def __init__(self,
                 dim,
                 heads=4,
                 att_drp=0.2,
                 feed_dim=3072,
                 feed_drp=0.2,
                 batch_first=True):
        # TODO 需要考虑加position embedding, 如果要用的话
        super().__init__()
        self.num_heads = heads
        self.layer = nn.ModuleDict({
            "att": nn.MultiheadAttention(dim, heads, att_drp, batch_first=batch_first),
            "add1": AddNormalBlock(dim),
            "feed": FeedForward(dim, feed_dim, dim, feed_drp),
            "add2": AddNormalBlock(dim),
        })

    def forward(self,
                input_seq,
                input_mask,
                condition_seq=None,
                condition_mask=None):
        # 这里自带的多头注意力机制, 本身就允许跨模态注意力机制
        condition_seq = input_seq if condition_seq is None else condition_seq

        attention_mask = (input_mask[:, :, None] * condition_mask[:, None, :])
        attention_mask = torch.repeat_interleave(attention_mask, repeats=self.num_heads, dim=0)
        # change the value of 1 to False
        condition_mask = (condition_mask == 1) == False
        attention_mask = (attention_mask == 1) == False

        enc_en, enc_en_att = self.layer["att"](query=input_seq,
                                               key=condition_seq,
                                               value=condition_seq,
                                            #    key_padding_mask=condition_mask,
                                               key_padding_mask=None,
                                               need_weights=True,
                                            #    attn_mask=attention_mask,
                                               attn_mask=None,
                                            #    average_attn_weights=True,
                                            #    is_causal=False
                                            )
        enc_en_temp = self.layer["add1"](enc_en, input_seq)
        enc_en = self.layer["feed"](enc_en_temp)
        enc_en = self.layer["add2"](enc_en, enc_en_temp)

        return enc_en


class CAEncoder(nn.Module):

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 attention_probs_dropout_prob,
                 hidden_dropout_prob,
                 intermediate_size,
                 num_hidden_layers,
                 batch_first=True):
        super().__init__()
        self.encoder = nn.ModuleDict({})
        for i in range(num_hidden_layers):
            self.encoder["layer_{}".format(i)] = EncoderLayer(
                hidden_size, num_attention_heads, attention_probs_dropout_prob,
                intermediate_size, hidden_dropout_prob, batch_first)

    def forward(self,
                input_seq,
                input_mask,
                condition_seq=None,
                condition_mask=None):
        x = input_seq
        for layer_name in self.encoder.keys():
            x = self.encoder[layer_name](x, input_mask, condition_seq, condition_mask)
        return x


class TGINet(nn.Module):

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_probs_dropout_prob,
        hidden_dropout_prob,
        intermediate_size,
        num_hidden_layers,
        position_embedding_type="absolute",
        max_position_embeddings=512,
        add_cross_attention=True,
        batch_first=True,
    ):
        super().__init__()
        self.add_cross_attention = add_cross_attention
        if position_embedding_type == "absolute":
            self.pos_emb = AbsolutePositionalEmbedding(
                hidden_size, max_position_embeddings)
        elif position_embedding_type == "sin":
            self.pos_emb = SinusoidalPositionEmbedding(hidden_size)
        else:
            self.pos_emb = nn.Identity()

        self.encoder = CAEncoder(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            batch_first=batch_first)
        self.pooler = MeanPooling()

    def forward(self, x, x_mask=None, condition_seq=None, condition_mask=None):
        pos_emb = self.pos_emb(x)  # [B, T, D]
        x_input = x + pos_emb
        if self.add_cross_attention:
            x_output = self.encoder(x_input,
                                    input_mask=x_mask,
                                    condition_seq=condition_seq,
                                    condition_mask=condition_mask)
        else:
            x_output = self.encoder(x_input, input_mask=x_mask)
        x_pooled = self.pooler(x_output, x_mask)
        return {"sequence": x_output, "embedding": x_pooled}
