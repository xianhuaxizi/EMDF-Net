"""
create by yjw on 2019-11-2
SGNN series model
"""

from utils import *


class GateGNN(Module):
    def __init__(self, hidden_size, dropout):
        super(GateGNN, self).__init__()
        self.hidden_size = hidden_size
        gate_size = hidden_size * 3

        self.b_a = Parameter(torch.empty(hidden_size), requires_grad=True)
        self.w = Parameter(torch.empty(gate_size, hidden_size), requires_grad=True)
        self.u = Parameter(torch.empty(gate_size, hidden_size), requires_grad=True)
        self.b_w = Parameter(torch.empty(gate_size), requires_grad=True)
        self.b_u = Parameter(torch.empty(gate_size), requires_grad=True)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        value = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-value, value)

    def cell(self, h, m):
        a = torch.matmul(m.transpose(1, 2), h) + self.b_a
        w = F.linear(a, self.w, self.b_w)
        u = F.linear(h, self.u, self.b_u)
        wz_a, wr_a, wc_a = w.chunk(3, 2)
        uz_h, ur_h, uc_h = u.chunk(3, 2)
        z = torch.sigmoid(wz_a + uz_h)
        r = torch.sigmoid(wr_a + ur_h)
        c = torch.tanh(wc_a + r * uc_h)
        h = c + z * (h - c)
        return self.dropout(h)

    def forward(self, hidden, matrix):
        outputs = self.cell(hidden, matrix)
        outputs = self.cell(outputs, matrix)
        return outputs


class Attention(Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

        self.w_i1 = nn.Linear(hidden_size, hidden_size // 2)
        self.w_i2 = nn.Linear(hidden_size // 2, 1)
        self.w_c1 = nn.Linear(hidden_size, hidden_size // 2)
        self.w_c2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, inputs):
        context = inputs[:, 0:8, :].repeat(1, 5, 1).view(-1, 8, self.hidden_size)
        candidate = inputs[:, 8:13, :]
        s_i = torch.relu(self.w_i1(context))
        s_i = torch.relu(self.w_i2(s_i))
        s_c = torch.relu(self.w_c1(candidate))
        s_c = torch.relu(self.w_c2(s_c))
        u = torch.tanh(torch.add(s_i.view(-1, 8), s_c.view(-1, 1)))
        a = (torch.exp(u) / torch.sum(torch.exp(u), 1).view(-1, 1)).view(-1, 8, 1)
        h_i = torch.sum(torch.mul(context, a), 1)
        h_c = (candidate / 8.0).view(-1, self.hidden_size)
        return h_i, h_c


class ScaleGNN(Module):
    def __init__(self, vocab_size, embedding_size, word_embedding, hidden_size,
                 dropout):
        super(ScaleGNN, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding.weight.data = torch.from_numpy(word_embedding)
        self.gate_gnn = GateGNN(hidden_size, dropout)
        self.attention = Attention(hidden_size)

    def get_params(self):
        model_grad_params = filter(lambda p: p.requires_grad, self.parameters())
        train_params = list(map(id, self.embedding.parameters()))
        tune_params = filter(lambda p: id(p) not in train_params, model_grad_params)
        return tune_params

    def forward(self, inputs, matrix):
        # embedding layer
        inputs_embed = self.embedding(inputs)
        inputs_embed = torch.cat(tuple([inputs_embed[:, i:i+13, :] for i in range(0, 52, 13)]), 2)

        # gate_gnn layer
        gnn_outputs = self.gate_gnn(inputs_embed, matrix)

        # attention layer
        h_i, h_c = self.attention(gnn_outputs)

        outputs = self.euclid(h_i, h_c)
        return outputs

    @staticmethod
    def predict(predict, label):
        _, predict = torch.sort(predict, descending=True)
        n_correct = torch.sum((predict[:, 0] == label)).item()
        n_label = label.size(0)
        acc = n_correct / n_label * 100.0
        return acc

    def predict_eval(self, inputs, matrix, label, set_index):
        predict = self.forward(inputs, matrix)
        for index in set_index:
            predict[index] = -1e9
        _, predict = torch.sort(predict, descending=True)
        n_correct = torch.sum((predict[:, 0] == label)).item()
        acc = n_correct / label.size(0) * 100.0
        predict = predict[:, 0:1].squeeze().cpu().numpy().tolist()
        label = label.cpu().numpy().tolist()
        predict_result = []
        for i in range(len(label)):
            if predict[i] == label[i]:
                predict_result.append(1)
            else:
                predict_result.append(0)
        return acc, predict_result

    @staticmethod
    def euclid(a, b):
        return -torch.norm(a - b, 2, 1).view(-1, 5)

