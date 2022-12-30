import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0):
        super(FeedForwardNetwork, self).__init__()
        self.dropout_rate = dropout_rate
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x_proj = F.dropout(F.relu(self.linear1(x)), p=self.dropout_rate, training=self.training)
        x_proj = self.linear2(x_proj)
        return x_proj


class PoolerStartLogits(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerStartLogits, self).__init__()
        self.dense = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, p_mask=None):
        x = self.dense(hidden_states)
        return x

class PoolerEndLogits(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerEndLogits, self).__init__()
        self.dense_0 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dense_1 = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, start_positions=None, p_mask=None):
        x = self.dense_0(torch.cat([hidden_states, start_positions], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x)
        return x

class Biaffine(nn.Module):
    def __init__(self, in_size, out_size, bias_x=True, bias_y=True):
        super().__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.out_size = out_size
        self.U = torch.nn.Parameter(torch.randn(in_size + int(bias_x),out_size,in_size + int(bias_y)))
        # self.U1 = self.U.view(size=(in_size + int(bias_x),-1))
        #U.shape = [in_size,out_size,in_size]  
    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), dim=-1)
        
        """
        batch_size,seq_len,hidden=x.shape
        bilinar_mapping=torch.matmul(x,self.U)
        bilinar_mapping=bilinar_mapping.view(size=(batch_size,seq_len*self.out_size,hidden))
        y=torch.transpose(y,dim0=1,dim1=2)
        bilinar_mapping=torch.matmul(bilinar_mapping,y)
        bilinar_mapping=bilinar_mapping.view(size=(batch_size,seq_len,self.out_size,seq_len))
        bilinar_mapping=torch.transpose(bilinar_mapping,dim0=2,dim1=3)
        """
        bilinar_mapping = torch.einsum('bxi,ioj,byj->bxyo', x, self.U, y)
        return bilinar_mapping