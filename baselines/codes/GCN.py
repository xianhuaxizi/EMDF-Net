"""
create by yjw on 2020-3-26
GCN series model
"""

from utils import *
from SGNN import ScaleGNN


class GCNModel(Module):
    def __init__(self, hidden_size, n_layers, dropout, in_out=False):
        super(GCNModel, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.in_out = in_out

        self.linear_list = nn.ModuleList([InitLinear(hidden_size, hidden_size, dis_func="normal", func_value=0.02)
                                          for _ in range(n_layers)])
        if in_out:
            self.linear = InitLinear(hidden_size*2, hidden_size, dis_func="normal", func_value=0.02)

    @staticmethod
    def normalize(a, x=None, in_out=False):
        if x is not None:
            n = torch.tanh(torch.matmul(x, x.permute(0, 2, 1)))
            a = a + torch.tanh(a * n)
            return a

        a = torch.add(a, torch.eye(a.size(1)).cuda())  # A+In
        d = torch.zeros_like(a)

        if in_out:
            d_in = torch.zeros_like(a)
            d_out = torch.zeros_like(a)
            d_i = torch.sum(a, 1)
            d_o = torch.sum(a, 2)
            for i in range(a.size(0)):
                d_in[i] = torch.diag(torch.pow(d_i[i], -0.5))
                d_out[i] = torch.diag(torch.pow(d_o[i], -0.5))
            return [torch.matmul(torch.matmul(d_in, a), d_in), torch.matmul(torch.matmul(d_out, a), d_out)]

        di = torch.sum(a, 2)
        for i in range(a.size(0)):
            d[i] = torch.diag(torch.pow(di[i], -0.5))
        return torch.matmul(torch.matmul(d, a), d)

    @staticmethod
    def masked(l, reverse=False):
        if not reverse:
            return torch.triu(l, 0)
        else:
            return torch.tril(l, 0)

    def forward(self, inputs, matrix):
        l = self.normalize(matrix, in_out=self.in_out)
        h = inputs

        if self.in_out:
            h_in = h
            h_out = h
            for linear in self.linear_list:
                h = torch.relu(linear(torch.matmul(l[0] + l[1], h)))
                h_in = torch.relu(linear(torch.matmul(l[0], h_in)))
                h_out = torch.relu(linear(torch.matmul(l[1], h_out)))
            h = self.linear(torch.cat([h_in, h_out], 2))
            return h

        for linear in self.linear_list:
            h = torch.relu(linear(torch.matmul(l, h)))
            h = nn.Dropout(self.dropout)(h)
        return h


class GCN(ScaleGNN):
    def __init__(self, n_layers, *args, **kwargs):
        super(GCN, self).__init__(*args, **kwargs)

        self.gcn = GCNModel(self.hidden_size, n_layers, self.dropout)

    def forward(self, inputs, matrix):
        # embedding layer
        inputs_embed = self.embedding(inputs)
        inputs_embed = torch.cat(tuple([inputs_embed[:, i:i+13, :] for i in range(0, 52, 13)]), 2)

        # GCN layer
        gcn_outputs = self.gcn(inputs_embed, matrix)

        # attention layer
        h_i, h_c = self.attention(gcn_outputs)

        outputs = self.euclid(h_i, h_c)
        return outputs


class ASAGCN(GCN):
    def __init__(self, positional_size, n_heads, *args, **kwargs):
        super(ASAGCN, self).__init__(*args, **kwargs)
        self.positional_size = positional_size

        self.arg_self_attention = SelfAttention(self.embedding_size, n_heads[0], self.dropout)
        self.layer_norm = nn.LayerNorm(self.embedding_size)
        self.event_composition = EventComposition(self.embedding_size, self.hidden_size, self.dropout)

    def adjust_event_chain_embedding(self, embedding):
        """  # 变换以后的tensor每batch_size个表示带有一个候选事件的事件链，这样可以形成多个事件链簇
        shape: (batch_size, 52, embedding_size) -> (batch_size * 5, 36, embedding_size)
            52: (8 context_event + 5 candidate event) * 4 arguments
            36: (8 context_event + 1 candidate event) * 4 arguments
        """
        embedding = torch.cat(tuple([embedding[:, i::13, :] for i in range(13)]), 1)  # 转换embedding的排列，变成以单独的每个事件的4个参数连续排列为主
        context_embedding = embedding[:, 0:32, :].repeat(1, 5, 1).view(-1, 32, self.embedding_size)
        candidate_embedding = embedding[:, 32:52, :].contiguous().view(-1, 4, self.embedding_size)
        event_chain_embedding = torch.cat((context_embedding, candidate_embedding), 1)
        return event_chain_embedding

    def adjust_event_embedding(self, embedding):
        """
        shape: (batch_size * 5, 9, hidden_size) -> (batch_size, 13, hidden_size)
        """
        embedding = embedding.view(embedding.size(0) // 5, -1, self.hidden_size)
        context_embedding = torch.zeros(embedding.size(0), 8, self.hidden_size).cuda()
        for i in range(0, 45, 9):
            context_embedding += embedding[:, i:i+8, :]
        context_embedding /= 8.0  # ???为什么出以8而不是5
        candidate_embedding = embedding[:, 8::9, :]
        event_embedding = torch.cat((context_embedding, candidate_embedding), 1)
        return event_embedding

    def forward(self, inputs, matrix):
        # embedding layer
        inputs_embed = self.embedding(inputs)
        inputs_embed = self.adjust_event_chain_embedding(inputs_embed)

        # argument attention layer
        mask = compute_mask(self.positional_size)
        arg_embed = self.arg_self_attention(inputs_embed, mask)
        arg_embed = self.layer_norm(torch.add(inputs_embed, arg_embed))

        # event composition layer
        event_embed = self.event_composition(arg_embed)
        event_embed = self.adjust_event_embedding(event_embed)

        # gcn layer
        gcn_outputs = self.gcn(event_embed, matrix)
        h_i, h_c = self.attention(gcn_outputs)

        # score functions
        # 1) Euclidean
        outputs = -torch.norm(h_i - h_c, 2, 1).view(-1, 5)
        # 2) Cosine
        # outputs = ((h_i / torch.norm(h_i, dim=1).view(-1, 1)) *
        #            (h_c / torch.norm(h_c, dim=1).view(-1, 1))).sum(-1).view(-1, 5)
        # 3) Dot
        # outputs = (h_i * h_c).sum(-1).view(-1, 5)
        # 4) Fusion
        # outputs = (torch.matmul(h_i, self.w1.view(-1, 1)) + torch.matmul(h_c, self.w2.view(-1, 1))).view(-1, 5)
        return outputs
