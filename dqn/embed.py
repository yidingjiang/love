import abc
import collections
import numpy as np
import torch
from torch import nn
from torch import distributions as td
from torch.nn import functional as F


class Embedder(abc.ABC, nn.Module):
    """Defines the embedding of an object in the forward method.

    Subclasses should register to the from_config method.
    """

    def __init__(self, embed_dim):
        """Sets the embed dim.

        Args:
            embed_dim (int): the dimension of the outputted embedding.
        """
        super().__init__()
        self._embed_dim = embed_dim

    @property
    def embed_dim(self):
        """Returns the dimension of the output (int)."""
        return self._embed_dim


class RecurrentStateEmbedder(Embedder):
    """Applies an LSTM on top of a state embedding."""

    def __init__(self, state_embedder, embed_dim):
        super().__init__(embed_dim)

        self._state_embedder = state_embedder
        self._lstm_cell = nn.LSTMCell(state_embedder.embed_dim, embed_dim)

    def forward(self, states, hidden_state=None):
        """Embeds a batch of sequences of contiguous states.

        Args:
            states (list[list[np.array]]): of shape
                (batch_size, sequence_length, state_dim).
            hidden_state (list[object] | None): batch of initial hidden states
                to use with the LSTM. During inference, this should just be the
                previously returned hidden state.

        Returns:
            embedding (torch.tensor): shape (batch_size, sequence_length, embed_dim)
            hidden_state (object): hidden state after embedding every element in the
                sequence.
        """
        batch_size = len(states)
        sequence_len = len(states[0])

        # Stack batched hidden state
        if batch_size > 1 and hidden_state is not None:
            hs = []
            cs = []
            for hidden in hidden_state:
                if hidden is None:
                    hs.append(torch.zeros(1, self.embed_dim))
                    cs.append(torch.zeros(1, self.embed_dim))
                else:
                    hs.append(hidden[0])
                    cs.append(hidden[1])
            hidden_state = (torch.cat(hs, 0), torch.cat(cs, 0))

        flattened = [state for seq in states for state in seq]

        # (batch_size * sequence_len, embed_dim)
        state_embeds, _ = self._state_embedder(flattened)
        state_embeds = state_embeds.reshape(batch_size, sequence_len, -1)

        embeddings = []
        for seq_index in range(sequence_len):
            hidden_state = self._lstm_cell(
                    state_embeds[:, seq_index, :], hidden_state)

            # (batch_size, 1, embed_dim)
            embeddings.append(hidden_state[0].unsqueeze(1))

        # (batch_size, sequence_len, embed_dim)
        # squeezed to (batch_size, embed_dim) if sequence_len == 1
        embeddings = torch.cat(embeddings, 1).squeeze(1)

        # Detach to save GPU memory.
        detached_hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())
        return embeddings, detached_hidden_state


import modules
class StateDependentEmbedder(Embedder):
    def __init__(self, embed_dim, feat_size=64):
        super().__init__(embed_dim)
        network_list = [modules.ConvLayer2D(input_size=3,
                                    output_size=feat_size,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1,
                                    normalize=False),  # 16 x 16
                        modules.ConvLayer2D(input_size=feat_size,
                                    output_size=feat_size,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1,
                                    normalize=False),  # 8 x 8
                        modules.ConvLayer2D(input_size=feat_size,
                                    output_size=feat_size,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1,
                                    normalize=False),  # 4 x 4
                        modules.ConvLayer2D(input_size=feat_size,
                                    output_size=feat_size,
                                    kernel_size=4,
                                    stride=1,
                                    padding=0,
                                    normalize=False),  # 1 x 1
                        modules.Flatten()]
        self.network_list = network_list
        network_list.append(
                modules.LinearLayer(input_size=feat_size, output_size=embed_dim))
        self.network = nn.Sequential(*network_list)

    def forward(self, input_data):
        #input_data = [torch.tensor(data) for data in input_data]
        # (B, H, W, C)
        input_data = torch.stack(input_data)
        # (B, C, H, W)
        input_data = input_data.permute(0, 3, 1, 2)
        # (B, embed_dim)
        #for layer in self.network_list:
        #    print(layer)
        #    input_data = layer(input_data)
        #    print(input_data.shape)

        #print(self.network_list[0](input_data))
        return self.network(input_data)

        #shape = list(input_data.shape) # (B, S, H, W, C)
        #input_data = input_data.reshape([-1] + shape[2:])  # (B x S, H, W, C)
        #input_data = input_data.permute(0, 3, 1, 2)  # (B x S, C, H, W)
        #out = self.network(input_data)
        #return out.reshape(shape[:2] + [-1])


class CompILEEmbedder(Embedder):
    def __init__(self, embed_dim):
        super().__init__(embed_dim)
        obj_embed_dim = 16
        self._encoder = modules.CompILEGridEncoder(output_dim=embed_dim)
        self._object_embedder = nn.Embedding(10, obj_embed_dim)
        self._final_layer = nn.Linear(embed_dim + obj_embed_dim, embed_dim)

    def forward(self, input_data, hidden_state=None):
        del hidden_state
        states, instruction = zip(*input_data)
        # (batch_size, W, H, C)
        states = torch.stack(states)

        # (batch_size,)
        instruction = torch.stack(instruction)

        # (batch_size, embed_dim)
        obj_embed = self._object_embedder(instruction)
        output = F.relu(self._encoder(states.unsqueeze(1))).squeeze(1)
        return self._final_layer(torch.cat((output, obj_embed), -1)), None


class World3DEmbedder(Embedder):
    def __init__(self, embed_dim):
        super().__init__(embed_dim)
        obj_embed_dim = 16
        self._encoder = modules.MiniWorldEncoderPano(output_dim=embed_dim)
        self._object_embedder = nn.Embedding(10, obj_embed_dim)
        self._final_layer = nn.Linear(embed_dim + obj_embed_dim, embed_dim)

    def forward(self, input_data, hidden_state=None):
        del hidden_state

        states, instruction = zip(*input_data)

        # (batch_size, W, H, C)
        states = torch.stack(states)

        # (batch_size,)
        instruction = torch.stack(instruction)

        # (batch_size, embed_dim)
        obj_embed = self._object_embedder(instruction)
        output = F.relu(self._encoder(states.unsqueeze(1))).squeeze(1)
        return self._final_layer(torch.cat((output, obj_embed), -1)), None
