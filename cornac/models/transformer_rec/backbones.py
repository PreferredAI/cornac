# Copyright 2026 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""HuggingFace backbone registry for :class:`TransformerRecModel`.

Each backbone is a bare HuggingFace transformer encoder that consumes
``inputs_embeds`` + ``attention_mask`` and exposes ``.last_hidden_state`` on
its output.
"""


def build_bert(vocab_size, embedding_dim, max_len, n_layers, n_heads, dropout, pad_idx):
    """Build a bidirectional BERT encoder (see BERT4Rec parameterization)."""
    from transformers.models.bert import BertConfig, BertModel

    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=embedding_dim,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        intermediate_size=embedding_dim * 4,
        hidden_act="gelu",
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        max_position_embeddings=max_len + 1,
        pad_token_id=pad_idx,
        layer_norm_eps=1e-12,
        use_cache=False,
    )
    return BertModel(config)


def build_gpt2(vocab_size, embedding_dim, max_len, n_layers, n_heads, dropout, pad_idx):
    """Build a causal GPT-2 decoder (see GPT2Rec parameterization)."""
    from transformers.models.gpt2 import GPT2Config, GPT2Model

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=max_len + 1,
        n_embd=embedding_dim,
        n_layer=n_layers,
        n_head=n_heads,
        n_inner=embedding_dim * 4,
        activation_function="gelu_new",
        resid_pdrop=dropout,
        embd_pdrop=dropout,
        attn_pdrop=dropout,
        pad_token_id=pad_idx,
        layer_norm_epsilon=1e-12,
        use_cache=False,
    )
    return GPT2Model(config)


def build_xlnet(vocab_size, embedding_dim, max_len, n_layers, n_heads, dropout, pad_idx):
    """Build an XLNet encoder (two-stream; supports perm_mask/target_mapping).

    ``d_head`` is set to ``embedding_dim // n_heads`` so that the attention
    output width matches ``d_model`` (the standard XLNet convention).
    """
    from transformers.models.xlnet import XLNetConfig, XLNetModel

    config = XLNetConfig(
        vocab_size=vocab_size,
        d_model=embedding_dim,
        n_layer=n_layers,
        n_head=n_heads,
        d_head=max(1, embedding_dim // n_heads),
        d_inner=embedding_dim * 4,
        ff_activation="gelu",
        dropout=dropout,
        pad_token_id=pad_idx,
        layer_norm_eps=1e-12,
        use_mems_train=False,
        use_mems_eval=False,
    )
    return XLNetModel(config)


def build_electra(vocab_size, embedding_dim, max_len, n_layers, n_heads, dropout, pad_idx):
    """Build an ELECTRA encoder (embedding_size and hidden_size both = dim)."""
    from transformers.models.electra import ElectraConfig, ElectraModel

    config = ElectraConfig(
        vocab_size=vocab_size,
        embedding_size=embedding_dim,
        hidden_size=embedding_dim,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        intermediate_size=embedding_dim * 4,
        hidden_act="gelu",
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        max_position_embeddings=max_len + 1,
        pad_token_id=pad_idx,
        layer_norm_eps=1e-12,
        use_cache=False,
    )
    return ElectraModel(config)


# name -> (build_fn, attention_type in {"causal", "bidirectional"})
BACKBONES = {
    "bert": (build_bert, "bidirectional"),
    "gpt2": (build_gpt2, "causal"),
    "xlnet": (build_xlnet, "bidirectional"),
    "electra": (build_electra, "bidirectional"),
}

# Read-only view mapping backbone name -> attention type (for validity checks).
ATTENTION_TYPES = {name: attn for name, (_, attn) in BACKBONES.items()}


def get_backbone(name):
    """Return the ``(build_fn, attention_type)`` pair for ``name``.

    Raises
    ------
    ValueError
        If ``name`` is not a registered backbone.
    """
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone '{name}'. Supported: {sorted(BACKBONES)}")
    return BACKBONES[name]
