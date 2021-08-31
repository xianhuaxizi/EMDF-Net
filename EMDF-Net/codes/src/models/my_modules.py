from src.util.tools import *


def compute_mask(positional_size):
    """
    Compute Mask matrix
    Mask: upper triangular matrix of masking subsequent information
        mask value: -1e9
    shape: (positional_size, positional_size)
    """
    return torch.triu(torch.fill_(torch.zeros(positional_size, positional_size), -1e9), 1)


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

# score functions
def calculate_related_score(h_i, h_c):
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

class InitLinear(Module):
    """
    Initialize Linear layer to be distribution function
    """
    def __init__(self, inputs_size, outputs_size, dis_func, func_value, bias=True):
        super(InitLinear, self).__init__()
        self.outputs_size = outputs_size

        self.weight = Parameter(torch.empty(inputs_size, outputs_size), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.empty(outputs_size), requires_grad=True)
        else:
            self.bias = None

        self.reset_parameters(dis_func, func_value)

    def reset_parameters(self, dis_func, func_value):
        if dis_func == "uniform":
            nn.init.uniform_(self.weight, -func_value, func_value)
            if self.bias is not None:
                nn.init.uniform_(self.bias, -func_value, func_value)

        if dis_func == "normal":
            nn.init.normal_(self.weight, std=func_value)
            if self.bias is not None:
                nn.init.normal_(self.bias, std=func_value)

    def forward(self, inputs):
        output_size = inputs.size()[:-1] + (self.outputs_size,)
        if self.bias is not None:
            outputs = torch.addmm(self.bias, inputs.view(-1, inputs.size(-1)), self.weight)
        else:
            outputs = torch.mm(inputs.view(-1, inputs.size(-1)), self.weight)
        outputs = outputs.view(*output_size)
        return outputs

class WeightedScoreSum(Module):
    """
    加权求和
    """
    def __init__(self, inputs_size):
        super(WeightedScoreSum, self).__init__()

        self.weight = Parameter(torch.empty(inputs_size), requires_grad=True)

        nn.init.ones_(self.weight)

    def forward(self, inputs):
        outputs = 0
        for i, input in enumerate(inputs):
            outputs += self.weight[i] * input
        return outputs

class SelfAttention(Module):
    """
    Self-Attention Layer

    Inputs:
        inputs: word embedding
        inputs.shape = (batch_size, sequence_length, embedding_size)

    Outputs:
        outputs: word embedding with context information
        outputs.shape = (batch_size, sequence_length, embedding_size)
    """
    def __init__(self, embedding_size, n_heads, dropout):
        super(SelfAttention, self).__init__()
        assert embedding_size % n_heads == 0
        self.embedding_size = embedding_size
        self.n_heads = n_heads
        self.d_k = embedding_size // n_heads

        self.w_qkv = InitLinear(embedding_size, embedding_size*3, dis_func="normal", func_value=0.02)
        self.w_head = InitLinear(embedding_size, embedding_size, dis_func="normal", func_value=0.02)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x, k=False):
        x = x.view(-1, x.size(1), self.n_heads, self.d_k)
        if k:
            return x.permute(0, 2, 3, 1)  # key.shape = (batch_size, n_heads, d_k, sequence_length)
        else:
            return x.permute(0, 2, 1, 3)  # query, value.shape = (batch_size, n_heads, sequence_length, d_k)

    def attention(self, query, key, value, mask=None):
        att = torch.matmul(query, key) / math.sqrt(self.d_k)

        if mask is not None:
            att = att + mask

        att = torch.softmax(att, -1)
        att = self.dropout(att)

        # att = self.sample(att)

        outputs = torch.matmul(att, value)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(-1, x.size(1), x.size(2) * x.size(3))
        x = self.dropout(self.w_head(x))
        return x

    @staticmethod
    def sample(att):
        att_ = att.view(-1, att.size(-1))
        _, tk = torch.topk(att_, k=26, largest=False)
        for i in range(att_.size(0)):
            att_[i][tk[i]] = 0.0
        att = att_.view(att.size())
        return att

    def forward(self, inputs, mask=None):
        inputs = self.w_qkv(inputs)
        query, key, value = torch.split(inputs, self.embedding_size, 2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)  # k=True，相当于进行了转置
        value = self.split_heads(value)

        att_outputs = self.attention(query, key, value, mask)
        outputs = self.merge_heads(att_outputs)
        return outputs

## MLP Attention
class StructuredSelfAttention(Module):
    def __init__(self, hidden_size, d_a, r):
        super(StructuredSelfAttention, self).__init__()

        self.d_a = d_a  # {int} hyperparameter,
        self.r = r      # {int} attention-hops or attention heads, num of event segments
        self.linear_first = nn.Linear(hidden_size, d_a)
        self.linear_first.bias.data.fill_(0)
        self.linear_second = nn.Linear(d_a, r)
        self.linear_second.bias.data.fill_(0)

    def forward(self, inputs):
        # inputs shape: [batch_size, seq_len, embedding_width]
        x = torch.tanh(self.linear_first(inputs))
        # x shape: [batch_size, seq_len, d_a]
        x = self.linear_second(x)
        # x shape: [batch_size, seq_len, r]
        x = F.softmax(x, 1)
        attention = x.transpose(1, 2)
        # att shape: [batch_size,  r, seq_len]
        segments_embeddings = attention @ inputs
        # output shape [batch_size, r, embedding_width]
        # avg_segments_embeddings = segments_embeddings.mean(1)

        return segments_embeddings, attention

class MulAttention(Module):
    def __init__(self, hidden_size, relu=False):
        super(MulAttention, self).__init__()

        self.w = nn.Linear(hidden_size, hidden_size)
        self.relu = relu

    def forward(self, inputs, mask=None):
        inputs_t = inputs.permute(0, 2, 1)
        u = torch.matmul(self.w(inputs), inputs_t)
        if self.relu:
            u = torch.relu(u)
        a = torch.softmax(u, 2)
        if mask is not None:
            a = a + mask
        h = torch.matmul(a, inputs)
        return h, a


class AddAttention(Module):
    def __init__(self, hidden_size, relu=False):
        super(AddAttention, self).__init__()

        self.hidden_size = hidden_size
        self.relu = relu

        self.w1 = nn.Linear(hidden_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)
        self.q = Parameter(torch.empty(hidden_size), requires_grad=True)
        nn.init.normal_(self.q, -0.02, 0.02)

    def forward(self, inputs, mask=None):
        arg_len = inputs.size(1)

        u1 = self.w1(inputs)
        u2 = self.w2(inputs)
        u = torch.zeros(inputs.size(0), arg_len, arg_len).to(inputs.device)
        for i in range(1, arg_len+1):
            u[:, i-1:i, :] = torch.matmul(torch.tanh(u1[:, i-1:i, :] + u2), self.q).view(-1, 1, arg_len)
        if self.relu:
            u = torch.relu(u)
        a = torch.softmax(u, 2)
        if mask is not None:
            a = a + mask
        h = torch.matmul(a, inputs)
        return h, a

class EventComposition(Module):
    """
    Event Composition layer
        integrate event argument embedding into event embedding
    Inputs:
        inputs: arguments embedding
        inputs.shape = (batch_size, argument_length, embedding_size)

    Outputs:
        outputs: event embedding
        outputs.shape = (batch_size, event_length, hidden_size)
    """
    def __init__(self, inputs_size, outputs_size, dropout):
        super(EventComposition, self).__init__()
        self.inputs_size = inputs_size

        self.w_e = InitLinear(inputs_size*4, outputs_size, dis_func="normal", func_value=0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        inputs = inputs.view(-1, inputs.size(1) // 4, self.inputs_size*4)
        outputs = self.dropout(torch.tanh(self.w_e(inputs)))
        return outputs

class EmbeddingFusion(Module):
    """
    Embedding Fusion layer
        integrate event embedding into event embedding
    Inputs:
        inputs: multi event embedding
        inputs.shape = (batch_size, event_length*n, hidden_size)

    Outputs:
        outputs: event embedding
        outputs.shape = (batch_size, event_length, hidden_size)
    """
    def __init__(self, inputs_size, outputs_size, dropout):
        super(EmbeddingFusion, self).__init__()

        self.linear1 = InitLinear(inputs_size, outputs_size, dis_func="normal", func_value=0.02)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        h1 = self.act(self.linear1(inputs))
        return self.dropout(h1)

# 以候选事件对上下文事件进行注意力。
class Attention(Module):
    def __init__(self, hidden_size, r=8):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

        self.w_i1 = nn.Linear(hidden_size, hidden_size // 2)
        self.w_i2 = nn.Linear(hidden_size // 2, 1)
        self.w_c1 = nn.Linear(hidden_size, hidden_size // 2)
        self.w_c2 = nn.Linear(hidden_size // 2, 1)
        self.r = r

    def forward(self, inputs):
        context = inputs[:, 0:self.r, :].repeat(1, 5, 1).view(-1, self.r, self.hidden_size)
        candidate = inputs[:, self.r::, :]
        s_i = torch.relu(self.w_i1(context))
        s_i = torch.relu(self.w_i2(s_i))
        s_c = torch.relu(self.w_c1(candidate))
        s_c = torch.relu(self.w_c2(s_c))
        u = torch.tanh(torch.add(s_i.view(-1, self.r), s_c.view(-1, 1)))
        a = (torch.exp(u) / torch.sum(torch.exp(u), 1).view(-1, 1)).view(-1, self.r, 1)
        h_i = torch.sum(torch.mul(context, a), 1)
        h_c = (candidate / self.r).view(-1, self.hidden_size)
        return h_i, h_c

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

        a = torch.add(a, torch.eye(a.size(1)).to(a.device))  # A+In
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