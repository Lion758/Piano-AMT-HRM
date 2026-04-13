"""
Converted jax-based code https://github.com/magenta/mt3/blob/main/mt3/layers.py#L489 to pytorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections.abc import Sequence

# device = "cuda" if torch.cuda.is_available() else "cpu"

def sinusoidal(shape, min_scale: float = 1.0,
               max_scale: float = 10000.0,
               dtype = torch.float32):
    """Sinusoidal init."""
    if dtype != torch.float32:
        raise ValueError('The sinusoidal initializer only supports float32.')
    if len(list(shape)) != 2:
        raise ValueError(
            f'Expected a 2D shape (max_len, features), but got {shape}.')
    max_len, features = shape
    pe = torch.zeros((max_len, features), dtype=dtype)
    position = torch.arange(0, max_len)[:, None]
    scale_factor = -np.log(max_scale / min_scale) / (features // 2 - 1)
    div_term = min_scale * torch.exp(torch.arange(0, features // 2) * torch.tensor(scale_factor))
    pe[:, :features // 2] = torch.sin(position * div_term)
    pe[:, features // 2:2 * (features // 2)] = torch.cos(position * div_term)
    
    return pe

class MlpBlock(nn.Module):
    def __init__(self,
                 emb_dim=512,
                 intermediate_dim=2048,
                 activations=('relu',),
                 kernel_init=nn.init.xavier_uniform_,
                 intermediate_dropout_rate=0.1,
                 dtype=torch.float32,
                 emb_dim_out = None
                 ):
        super(MlpBlock, self).__init__()
        self.dtype = dtype
        self.intermediate_layers = nn.ModuleList([nn.Linear(in_features=emb_dim, out_features=intermediate_dim), nn.ReLU()])
        self.dropout = nn.Dropout(intermediate_dropout_rate)
        if emb_dim_out is None:
            emb_dim_out = emb_dim
        self.dense_layer = nn.Linear(in_features=intermediate_dim, out_features=emb_dim_out)

    def forward(self, inputs):
        x = inputs

        for layer in self.intermediate_layers:
            x = layer(x)

        x = self.dropout(x)
        output = self.dense_layer(x)
        return output


def normalize_transformer_ffn_activation(activations):
    if activations is None:
        return "relu"

    if isinstance(activations, str):
        activation_name = activations
    elif isinstance(activations, Sequence):
        if len(activations) != 1:
            raise ValueError(f"Expected a single FFN activation, got {activations!r}")
        activation_name = activations[0]
    else:
        activation_name = str(activations)

    activation_name = str(activation_name).strip().lower()
    if activation_name not in {"relu", "swiglu"}:
        raise ValueError(f"Unsupported transformer FFN activation: {activation_name}")
    return activation_name


class TransformerFfnBlock(nn.Module):
    def __init__(self,
                 emb_dim=512,
                 intermediate_dim=2048,
                 activations=("relu",),
                 intermediate_dropout_rate=0.1,
                 emb_dim_out=None,
                 ):
        super(TransformerFfnBlock, self).__init__()
        self.activation_name = normalize_transformer_ffn_activation(activations)
        self.dropout = nn.Dropout(intermediate_dropout_rate)
        if emb_dim_out is None:
            emb_dim_out = emb_dim

        if self.activation_name == "relu":
            # Preserve legacy module names so existing ReLU checkpoints keep loading.
            self.intermediate_layers = nn.ModuleList([
                nn.Linear(in_features=emb_dim, out_features=intermediate_dim),
                nn.ReLU(),
            ])
        else:
            self.value_layer = nn.Linear(in_features=emb_dim, out_features=intermediate_dim)
            self.gate_layer = nn.Linear(in_features=emb_dim, out_features=intermediate_dim)
            self.silu = nn.SiLU()

        self.dense_layer = nn.Linear(in_features=intermediate_dim, out_features=emb_dim_out)

    def forward(self, inputs):
        if self.activation_name == "relu":
            x = inputs
            for layer in self.intermediate_layers:
                x = layer(x)
        else:
            x = self.value_layer(inputs) * self.silu(self.gate_layer(inputs))

        x = self.dropout(x)
        return self.dense_layer(x)

class Embed(nn.Module):
    def __init__(self,
                num_embeddings,
                features,
                dtype=torch.float32,
                attend_dtype=None,
                embedding_init=nn.init.normal_,
                one_hot=False):
        super(Embed, self).__init__()
        self.num_embeddings = num_embeddings
        self.dtype = dtype
        self.embedding_init = embedding_init
        self.one_hot = one_hot
        # nn.Embedding(num_embeddings, embedding_dim=features, padding_idx=)
        self.embedding = nn.Parameter(torch.Tensor(num_embeddings, features)) #.to(device) (if use .to(device), this embedding will not be saved when saving state_dict to checkpoint.)
        self.reset_parameters()

    def reset_parameters(self):
        self.embedding_init(self.embedding)

    def forward(self, inputs):
        if self.one_hot:
            iota = torch.arange(self.num_embeddings, dtype=torch.int32).to(inputs.device)
            one_hot = (inputs[..., None] == iota).type(self.dtype)
            output = torch.matmul(one_hot, self.embedding)
        else:
            output = self.embedding[inputs]

        return output

class FixedEmbed(nn.Module):
    def __init__(self, features, max_length=2048, dtype=torch.float32):
        super(FixedEmbed, self).__init__()
        self.features = features
        self.max_length = max_length
        self.dtype = dtype
        embeding = sinusoidal(shape=(self.max_length, self.features), dtype=self.dtype)
        self.register_buffer("embedding", embeding)

    def forward(self, inputs, decode=False):
        # if decode:
        #     position_embedder_index = self.variable(
        #         'cache', 'position_embedder_index',
        #         lambda: torch.array(-1, dtype=torch.uint32).to(device)
        #     )
        #     i = position_embedder_index.value
        #     position_embedder_index.value = i + 1

        #     return self.embedding[i:i+1, :]

        return self.embedding[inputs, :]

#Referenced https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/models/t5/modeling_t5.py#L238
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        #convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states
