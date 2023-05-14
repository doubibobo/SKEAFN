import dgl
import numpy
import torch
import torch.nn as nn

from ca import TGINet
from fafw import FAFWNet
from classifier import LogisticModel
from transformers import BertModel, RobertaModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TextME(nn.Module):

    def __init__(self, model_name="bert-base-uncased", use_finetune=True):
        super(TextME, self).__init__()
        self.model_name = model_name
        self.is_roberta = "roberta" in model_name
        if self.is_roberta:
            self.model = RobertaModel.from_pretrained(model_name,
                                                      cache_dir=None)
        else:
            self.model = BertModel.from_pretrained(model_name, cache_dir=None)
        self.use_finetune = use_finetune

    def forward(self, ids, attention_mask, segment_ids):
        if self.use_finetune:
            if self.is_roberta:
                outputs = self.model(input_ids=ids, attention_mask=attention_mask)
            else:
                outputs = self.model(input_ids=ids, attention_mask=attention_mask, token_type_ids=segment_ids)
        else:
            with torch.no_grad():
                if self.is_roberta:
                    outputs = self.model(input_ids=ids, attention_mask=attention_mask)
                else:
                    outputs = self.model(input_ids=ids, attention_mask=attention_mask, token_type_ids=segment_ids)
        return {
            "sequence": outputs[0],
            "embedding": outputs[1],
        }


class RNNME(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 number_layers=1,
                 drop_rate=0.2,
                 bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional

        self.rnn = nn.LSTM(input_size,
                           hidden_size,
                           num_layers=number_layers,
                           dropout=drop_rate,
                           bidirectional=bidirectional,
                           batch_first=True)

        self.layer_norm = nn.LayerNorm(
            ((2 if bidirectional else 1) * hidden_size))

        self.dropout = nn.Dropout(drop_rate)
        self.linear_1 = nn.Linear((2 if bidirectional else 1) * hidden_size,
                                  output_size)
        self.linear_2 = nn.Linear((2 if bidirectional else 1) * hidden_size,
                                  output_size)

    def forward(self, x, mask, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        mask: 与其它两个encode统一而添加, 不具备实际含义
        '''
        lengths = lengths.to(torch.int64).to("cpu")
        bs, seq_ln, _ = x.size()

        packed_sequence = pack_padded_sequence(x,
                                               lengths,
                                               batch_first=True,
                                               enforce_sorted=False)
        lstm_outputs, final_states = self.rnn(packed_sequence)

        lstm_outputs, _ = pad_packed_sequence(lstm_outputs,
                                              batch_first=True,
                                              total_length=seq_ln)
        lstm_outputs = self.layer_norm(lstm_outputs)

        if self.bidirectional:
            h = self.dropout(
                torch.cat((final_states[0][0], final_states[0][1]), dim=-1))
        else:
            h = self.dropout(final_states[0].squeeze())

        y_1 = self.linear_1(h)
        lstm_outputs = self.linear_2(lstm_outputs)
        return {
            "sequence": lstm_outputs,
            "embedding": y_1,
        }


class TextGNN(nn.Module):

    def __init__(self,
                 node_hidden_size,
                 vocab_path,
                 edge_weights_path,
                 edge_matrix_path,
                 vocab_embedding_path,
                 edge_trainable=False,
                 graph_embedding_drop_rate=0.1):
        """
        This is the note
        :param node_hidden_size:
        :param vocab_path:
        # :param max_length: this param will be determined in the dataset loader and config file, not need here
        :param edge_matrix_path:
        :param vocab_embedding_path: dict, key is the word, and value is the embedding_vector
        :param edge_trainable:
        :param graph_embedding_drop_rate:
        """
        super(TextGNN, self).__init__()
        self.node_hidden_size = node_hidden_size

        def load_vocab_edge_message(vocab, edge_weights, edge_matrix, vocab_embedding):
            vocab = numpy.load(vocab, allow_pickle=True).item()
            edge_matrix = numpy.load(edge_matrix)
            edge_weights = numpy.load(edge_weights)
            vocab_bert_embedding = numpy.load(vocab_embedding, allow_pickle=True).item()
            return len(edge_weights), edge_weights, edge_matrix, vocab, vocab_bert_embedding

        self.edge_number, self.edge_weights, self.edge_matrix, self.vocab, self.vocab_embedding = load_vocab_edge_message(
            vocab=vocab_path,
            edge_weights=edge_weights_path,
            edge_matrix=edge_matrix_path,
            vocab_embedding=vocab_embedding_path)

        self.node_hidden = torch.nn.Embedding(len(self.vocab), node_hidden_size)
        self.node_hidden.weight.data.copy_(torch.tensor(self.load_vocab_embedding()))
        self.node_hidden.weight.requires_grad = True

        if edge_trainable:
            self.edge_hidden = torch.nn.Embedding.from_pretrained(torch.ones((self.edge_number, 1)), freeze=False)
        else:
            self.edge_hidden = torch.nn.Embedding.from_pretrained(torch.tensor(self.edge_weights, dtype=torch.float32), freeze=False)

        self.node_eta = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.node_eta.data.fill_(0)

        self.embedding_layer1 = nn.Linear(768, 768)
        self.dropout_layer = nn.Dropout(p=graph_embedding_drop_rate)
        self.activation_layer = nn.Sigmoid()

    def load_vocab_embedding(self):
        # 这里要保证vocab embedding和word to id之间的对应关系
        vocab_embedding_matrix = numpy.zeros(
            (len(self.vocab), self.node_hidden_size))
        for word, word_id in self.vocab.items():
            vocab_embedding_matrix[word_id] = self.vocab_embedding[word]
        print('THE SHAPE OF EMBEDDING MATRIX IS {}.\n'.format(
            vocab_embedding_matrix.shape))
        return vocab_embedding_matrix

    def create_graph_for_a_sample(self, token_id):
        """
        为样本构建图
        :param token_id: shape is <T>. Type is numpy.
        :return:
        """
        # if len(token_id) > self.max_length_per_sample:
        #     # TODO 数据加载时即可完成，这里不需要做额外处理
        #     pass
        local_vocab_id = set(token_id)
        old_to_new_vocab_id = dict(
            zip(local_vocab_id, range(len(local_vocab_id))))  # 从旧ID到新ID的转换

        # 1. 构建子图
        # sub_graph = dgl.DGLGraph().to(self.node_eta.device)
        sub_graph = dgl.graph(([], [])).to(self.node_eta.device)

        # 2. 构建节点
        sub_graph.add_nodes(len(local_vocab_id))  # 构建与文本长度大小一致的节点
        sub_graph.ndata['h'] = self.node_hidden(
            torch.Tensor(list(local_vocab_id)).int().to(self.node_eta.device))

        # 3. 构建边
        edges, old_edge_id = self.create_edges_for_a_sample(
            local_vocab_id, old_to_new_vocab_id)
        id_src, id_dst = zip(*edges)
        sub_graph.add_edges(id_src, id_dst)
        sub_graph.edata['w'] = self.edge_hidden(
            torch.Tensor(list(old_edge_id)).int().to(self.node_eta.device))

        return sub_graph

    def create_edges_for_a_sample(self, token_id, old_to_new_vocab_id):
        """
        为样本构建边，如何定义边是否存在？
        因为这里直接是构建子图，所以需要将大图节点信息转化为子图节点信息
        :param token_id: 样本数据ID
        :param old_to_new_vocab_id: 全局字典转化为局部字典
        :return:
            edges: 子图的边信息（使用子图的节点信息进行展示），格式为[new_id, new_id]
            old_edge_id: 子图边ID信息 (使用大图的节点ID进行展示), 格式为[message_for_old_edge_id]
        """
        edges = []
        old_edge_id = []
        new_token_id = []

        # TODO 如何处理全部为padding的情况
        for item_id in token_id:
            if item_id != 0:
                new_token_id.append(item_id)
            else:
                pass
        if len(new_token_id) == 0:
            new_token_id.append(0)

        for index, word_old_id in enumerate(new_token_id):
            new_token_id_src = old_to_new_vocab_id[word_old_id]
            for i in range(len(new_token_id)):
                new_token_id_dst = old_to_new_vocab_id[new_token_id[i]]
                # 新建一条边，已经包括了自环
                edges.append([new_token_id_src, new_token_id_dst])
                old_edge_id.append(self.edge_matrix[word_old_id, new_token_id[i]])

        return edges, old_edge_id

    def forward(self, token_ids):
        """
        图的输入：每个样本中可能含有的情感词，可能需要提前补零
        :param token_ids: shape is <B, T>. B and T denotes batch size and text length respectively. Type is torch.Tensor
        :return:
        """
        token_ids = token_ids.cpu().numpy().tolist()
        sub_graphs = [
            self.create_graph_for_a_sample(token_id) for token_id in token_ids
        ]

        # TODO 这里需要改成单张图的信息传递过程

        # 1. 初始化大图
        batch_graph = dgl.batch(sub_graphs).to(self.node_eta.device)
        before_node_embedding = batch_graph.ndata['h']

        # 2. 完成大图的更新
        batch_graph.update_all(
            message_func=dgl.function.src_mul_edge('h', 'w', 'weighted_message'),
            reduce_func=dgl.function.max('weighted_message', 'h'))

        after_node_embedding = batch_graph.ndata['h']

        # 3. 处理聚合后和聚合前图的特征信息
        new_node_embedding = self.node_eta * before_node_embedding + (1 - self.node_eta) * after_node_embedding
        batch_graph.ndata['h'] = new_node_embedding

        # 计算整张图的特征
        graph_embedding = dgl.mean_nodes(batch_graph, feat='h')
        graph_embedding = self.embedding_layer1(graph_embedding)
        graph_embedding = self.activation_layer(graph_embedding)
        graph_embedding = self.dropout_layer(graph_embedding)
        # graph_embedding = self.norm_layer(graph_embedding)
        # graph_embedding = self.embedding_layer2(graph_embedding)
        return graph_embedding


class SKEAFN(nn.Module):

    def __init__(self, model_config) -> None:
        super().__init__()

        self.input_channels = model_config["input_channels"]

        self.me_text = nn.ModuleList([
            TextME(**model_config["me_text"]["pre"]["params"]),
            RNNME(**model_config["me_text"]["post"]["params"]),
        ])
        self.me_acoustic = RNNME(**model_config["me_acoustic"]["params"])
        self.me_visual = RNNME(**model_config["me_visual"]["params"])
        self.eke = TextGNN(**model_config["eke"]["params"])

        self.tgi_at = TGINet(**model_config["fusion"]["pre"]["params"])
        self.tgi_vt = TGINet(**model_config["fusion"]["pre"]["params"])

        model_config["fusion"]["post"]["params"]["input_dim"] = int(
            model_config["fusion"]["post"]["params"]["input_dim"] *
            sum(list(self.input_channels.values())))
        self.fafw = FAFWNet(**model_config["fusion"]["post"]["params"])

        model_config["classifier"]["params"]["input_dim"] = int(
            model_config["classifier"]["params"]["input_dim"] *
            sum(list(self.input_channels.values())))
        self.classifier = LogisticModel(**model_config["classifier"]["params"])

    def forward(self, input_dict):
        prob_dict = {}

        # 给定pth文件之后的测试模块, input_channels并没有被保存, 需要进行处理
        if not hasattr(self, "input_channels"):
            self.input_channels = {
                "text": True,
                "acoustic": True,
                "visual": True,
                "kb": True,
            }

        if self.input_channels["text"]:
            text_ouput_dict = self.me_text[0](input_dict["text"],
                                              input_dict["text_mask"],
                                              input_dict["text_segment_ids"])
            text_ouput_dict = self.me_text[1](text_ouput_dict["sequence"],
                                              input_dict["text_mask"],
                                              input_dict["text_length"])
            text_sequence = text_ouput_dict["sequence"]
            text_embedding = text_ouput_dict["embedding"]

        if self.input_channels["acoustic"]:
            acoustic_output_dict = self.me_acoustic(input_dict["acoustic"], 
                                                    input_dict["acoustic_mask"],
                                                    input_dict["acoustic_length"])
            acoustic_sequence = acoustic_output_dict["sequence"]
            acoustic_embedding = acoustic_output_dict["embedding"]

        # ocr_embedding = ocr_head_outputs['sequence']
        # ocr_feature = ocr_head_outputs['embedding']

        if self.input_channels["visual"]:
            visual_output_dict = self.me_visual(input_dict["visual"],
                                                input_dict["visual_mask"],
                                                input_dict["visual_length"])
            visual_sequence = visual_output_dict["sequence"]
            visual_embedding = visual_output_dict["embedding"]

        if self.input_channels["kb"]:
            kb_embedding = self.eke(input_dict["kb"])

        if self.input_channels["text"] and self.input_channels["visual"]:
            text_visual_dict = self.tgi_vt(text_sequence,
                                           input_dict["text_mask"],
                                           visual_sequence,
                                           input_dict["visual_mask"])
            text_visual_sequence = text_visual_dict["sequence"]
            text_visual_embedding = text_visual_dict["embedding"]

        if self.input_channels["text"] and self.input_channels["acoustic"]:
            text_acoustic_output_dict = self.tgi_at(text_sequence,
                                                    input_dict["text_mask"],
                                                    acoustic_sequence,
                                                    input_dict["acoustic_mask"])
            text_acoustic_sequence = text_acoustic_output_dict["sequence"]
            text_acoustic_embedding = text_acoustic_output_dict["embedding"]

        if sum(list(self.input_channels.values())) == 4:
            fusion_feature = torch.cat(
                [
                    text_embedding, text_visual_embedding,
                    text_acoustic_embedding, kb_embedding
                ],
                dim=1,
            )
        elif sum(list(self.input_channels.values())) == 3:
            if (self.input_channels["text"] and self.input_channels["visual"]
                    and self.input_channels["acoustic"]):
                fusion_feature = torch.cat([
                    text_embedding, text_visual_embedding,
                    text_acoustic_embedding
                ],
                                           dim=1)
            elif (self.input_channels["text"] and self.input_channels["visual"]
                  and self.input_channels["kb"]):
                fusion_feature = torch.cat(
                    [text_embedding, text_visual_embedding, kb_embedding],
                    dim=1,
                )
            elif (self.input_channels["text"]
                  and self.input_channels["acoustic"]
                  and self.input_channels["kb"]):
                fusion_feature = torch.cat(
                    [text_embedding, text_acoustic_embedding, kb_embedding],
                    dim=1,
                )
            elif (self.input_channels["visual"]
                  and self.input_channels["acoustic"]
                  and self.input_channels["kb"]):
                fusion_feature = torch.cat(
                    [visual_embedding, acoustic_embedding, kb_embedding],
                    dim=1,
                )
            else:
                # 不可能情形
                pass
        elif sum(list(self.input_channels.values())) == 2:
            # 双模态情形，共六种
            if self.input_channels["text"] and self.input_channels["visual"]:
                fusion_feature = torch.cat(
                    [text_embedding, text_visual_embedding],
                    dim=1,
                )
            elif self.input_channels["text"] and self.input_channels[
                    "acoustic"]:
                fusion_feature = torch.cat(
                    [text_embedding, text_visual_embedding],
                    dim=1,
                )
            elif self.input_channels["text"] and self.input_channels["kb"]:
                fusion_feature = torch.cat(
                    [text_embedding, kb_embedding],
                    dim=1,
                )
            elif self.input_channels["visual"] and self.input_channels[
                    "acoustic"]:
                fusion_feature = torch.cat(
                    [visual_embedding, acoustic_embedding],
                    dim=1,
                )
            elif self.input_channels["visual"] and self.input_channels["kb"]:
                fusion_feature = torch.cat(
                    [visual_embedding, kb_embedding],
                    dim=1,
                )
            elif self.input_channels["acoustic"] and self.input_channels["kb"]:
                fusion_feature = torch.cat(
                    [acoustic_embedding, kb_embedding],
                    dim=1,
                )
            else:
                # 不可能情形
                pass
        elif sum(list(self.input_channels.values())) == 1:
            # 单模态情形，共四种
            if self.input_channels["text"]:
                fusion_feature = text_embedding
            elif self.input_channels["visual"]:
                fusion_feature = visual_embedding
            elif self.input_channels["acoustic"]:
                fusion_feature = acoustic_embedding
            elif self.input_channels["kb"]:
                fusion_feature = kb_embedding
            else:
                # 不可能情形
                pass
        else:
            # 不可能情形
            pass

        fusion_dict = self.fafw(fusion_feature)
        prob_dict = self.classifier(fusion_dict["output"])
        return prob_dict
