'''
(c) Pengpeng Zhou
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.util.tools import *
from .my_modules import *

class FusionNet(Module):
    def __init__(self, vocab_size, embedding_size, word_embedding, hidden_size, dropout,
                 num_layers, bidirectional, positional_size, n_heads, d_a, r):
        super(FusionNet, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.positional_size = positional_size
        self.n_heads = n_heads
        num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding.weight.data = torch.from_numpy(word_embedding)

        #  event-level context representation
        self.event_level_block = nn.ModuleList()
        self.event_level_block.add_module('arg_self_attention', SelfAttention(self.embedding_size, n_heads[0], self.dropout))
        self.event_level_block.add_module('layer_norm', nn.LayerNorm(self.embedding_size))
        self.event_level_block.add_module('event_composition', EventComposition(self.embedding_size, self.hidden_size, self.dropout))
        self.event_level_block.add_module('attention', Attention(hidden_size))

        #  chain-level context representation
        self.chain_level_block = nn.ModuleList()
        self.chain_level_block.add_module('gcn', GCNModel(self.hidden_size, num_layers, self.dropout))
        self.chain_level_block.add_module('attention', Attention(hidden_size))

        # segment-level context representation
        self.segment_level_block = nn.ModuleList()
        # self.segment_level_block.add_module('mul_attention_segment',MulAttention(self.hidden_size, relu=True))
        # self.segment_level_block.add_module('add_attention_segment', AddAttention(self.hidden_size, relu=True))
        # MLP Attention
        self.segment_level_block.add_module('stru_attention_segment', StructuredSelfAttention(self.hidden_size, d_a, r))
        self.segment_level_block.add_module('attention', Attention(hidden_size, r))

        # feature fusion layer
        self.fusion_level_block = nn.ModuleList()
        self.fusion_level_block.add_module('contextFusion', EmbeddingFusion(self.hidden_size*3, self.hidden_size, self.dropout) )
        self.fusion_level_block.add_module('candidateFusion',EmbeddingFusion(self.hidden_size*2, self.hidden_size, self.dropout))

        # score fusion layer
        self.scoreFusion = WeightedScoreSum(4)

    def adjust_event_chain_embedding_type1(self, embedding):
        """
        shape: (batch_size, 52, embedding_size) -> (batch_size * 5, 9, hidden_size)
            52: (8 context_event + 5 candidate event) * 4 arguments
            9: (8 context_event + 1 candidate event)
        """
        embedding = torch.cat(tuple([embedding[:, i:i+13, :] for i in range(0, 52, 13)]), 2)
        context_embedding = embedding[:, 0:8, :].repeat(1, 5, 1).view(-1, 8, self.hidden_size)
        candidate_embedding = embedding[:, 8:13, :].contiguous().view(-1, 1, self.hidden_size)
        event_chain_embedding = torch.cat((context_embedding, candidate_embedding), 1)
        return event_chain_embedding

    def adjust_event_chain_embedding_type2(self, embedding):
        """
        shape: (batch_size, 52, embedding_size) -> (batch_size * 5, 36, embedding_size)
            52: (8 context_event + 5 candidate event) * 4 arguments
            36: (8 context_event + 1 candidate event) * 4 arguments
        """
        embedding = torch.cat(tuple([embedding[:, i::13, :] for i in range(13)]), 1)
        context_embedding = embedding[:, 0:32, :].repeat(1, 5, 1).view(-1, 32, self.embedding_size)
        candidate_embedding = embedding[:, 32:52, :].contiguous().view(-1, 4, self.embedding_size)
        event_chain_embedding = torch.cat((context_embedding, candidate_embedding), 1)
        return event_chain_embedding

    def adjust_event_embedding(self, embedding):
        """
        shape: (batch_size * 5, 9, hidden_size) -> (batch_size, 13, hidden_size)
        """
        embedding = embedding.view(embedding.size(0) // 5, -1, self.hidden_size)
        context_embedding = torch.zeros(embedding.size(0), 8, self.hidden_size).to(embedding.device)
        for i in range(0, 45, 9):
            context_embedding += embedding[:, i:i+8, :]
        context_embedding /= 8.0  # ???为什么出以8而不是5
        candidate_embedding = embedding[:, 8::9, :]
        event_embedding = torch.cat((context_embedding, candidate_embedding), 1)
        return event_embedding

    def get_params(self):
        model_grad_params = filter(lambda p: p.requires_grad, self.parameters())
        train_params = list(map(id, self.embedding.parameters()))
        tune_params = filter(lambda p: id(p) not in train_params, model_grad_params)
        return tune_params

    def forward(self, inputs, matrix):
        # embedding layer
        inputs_embed = self.embedding(inputs)

        # event-level
        inputs_embed = self.adjust_event_chain_embedding_type2(inputs_embed)
        mask = compute_mask(self.positional_size[0]).to(inputs_embed.device)
        arg_embed =self.event_level_block.arg_self_attention(inputs_embed, mask)
        arg_embed = self.event_level_block.layer_norm(torch.add(inputs_embed, arg_embed))
        event_embed_level1 = self.event_level_block.event_composition(arg_embed)

        #  event-level context representation
        event_embed_level1_all = self.adjust_event_embedding(event_embed_level1)
        h_i_level1, h_c_level1 = self.event_level_block.attention(event_embed_level1_all)

        # chain-level
        event_embed_level2_all = self.chain_level_block.gcn(event_embed_level1_all, matrix)
        h_i_level2, h_c_level2 = self.chain_level_block.attention(event_embed_level2_all)

        # segment-level
        context = event_embed_level2_all[:, 0:8, :].contiguous()
        # context_seg, att = self.segment_level_block.add_attention_segment(context)
        # context_seg, att = self.segment_level_block.mul_attention_segment(context)
        # att = None

        # MLP Attention
        context_seg, att = self.segment_level_block.stru_attention_segment(context)
        candidate = event_embed_level2_all[:, 8:13, :].contiguous()
        event_embed_level3_all = torch.cat((context_seg, candidate), 1)
        h_i_level3, h_c_level3 = self.segment_level_block.attention(event_embed_level3_all)  # h_c_level3 与 h_c_level2 是相等的，来自同一embedding


        # feature fusion
        h_i_level_fusion = torch.cat([h_i_level1, h_i_level2, h_i_level3], 1)
        h_i_level_fusion = self.fusion_level_block.contextFusion(h_i_level_fusion)
        h_c_level_fusion = torch.cat([h_c_level1, h_c_level2], 1)
        h_c_level_fusion = self.fusion_level_block.candidateFusion(h_c_level_fusion)

        # prediction score
        outputs_level1 = calculate_related_score(h_i_level1, h_c_level1)
        outputs_level2 = calculate_related_score(h_i_level2, h_c_level2)
        outputs_level3 = calculate_related_score(h_i_level3, h_c_level3)
        outputs_level4 = calculate_related_score(h_i_level_fusion, h_c_level_fusion)

        # score fusion
        outputs_fusion = self.scoreFusion([outputs_level1, outputs_level2, outputs_level3, outputs_level4])

        return [outputs_level1, outputs_level2, outputs_level3, outputs_level4, outputs_fusion, att]
