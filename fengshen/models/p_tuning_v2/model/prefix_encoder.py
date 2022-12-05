import torch
import pdb

class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        self.pre_seq_len = config.pre_seq_len
        self.alpha = 0.01
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
            )
            # self.trans = torch.nn.Sequential(
            #     torch.nn.Linear(config.hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
            # )
        else:
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.num_hidden_layers * 2 * config.hidden_size)
            p_param=0
            for name, param in self.embedding.named_parameters():
                p_param += param.numel()
            print('p param is {}'.format(p_param))
            self.knowledge_trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size))
            # self.relation_trans = torch.nn.Sequential(
            #     torch.nn.Linear(config.num_hidden_layers * 2 * config.hidden_size, config.hidden_size)
            # )

    def forward(self, prefix, knowledge_embeddings=None, relation_embeddins=None):
        # pdb.set_trace()
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
            if knowledge_embeddings!= None:
                # pdb.set_trace()
                knowledge_embeddings=knowledge_embeddings.repeat(past_key_values.size(0), 1, 1)
                knowledge_past_key_values = self.knowledge_trans(knowledge_embeddings)
                # pdb.set_trace()
                past_key_values = past_key_values + knowledge_past_key_values*self.alpha
                print("using knowledge trans")
                # pdb.set_trace()
                # pdb.set_trace()
                # past_key_values = torch.cat([past_key_values, knowledge_past_key_values], axis=1)
        return past_key_values


class ExplPrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, config):
        super().__init__()
        self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
            )
        # self.prefix_projection = config.prefix_projection
        # if self.prefix_projection:
        #     # Use a two-layer MLP to encode the prefix
        #     self.embedding = torch.nn.Embedding(config.pre_seq_len, config.hidden_size)
        #     self.trans = torch.nn.Sequential(
        #         torch.nn.Linear(config.hidden_size, config.prefix_hidden_size),
        #         torch.nn.Tanh(),
        #         torch.nn.Linear(config.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
        #     )
        # else:
        #     self.embedding = torch.nn.Embedding(config.pre_seq_len, config.num_hidden_layers * 2 * config.hidden_size)

    def forward(self, nle_hidden_states: torch.Tensor):
        past_key_values = self.trans(nle_hidden_states)
        # if self.prefix_projection:
        #     prefix_tokens = self.embedding(prefix)
        #     past_key_values = self.trans(prefix_tokens)
        # else:
        #     past_key_values = self.embedding(prefix)
        return past_key_values