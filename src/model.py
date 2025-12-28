import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer architecture"""
    
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len
        self._generate_pe(max_len)
    
    def _generate_pe(self, max_len):
        """Generate positional encoding matrix"""
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Forward pass with dynamic extension if needed"""
        seq_len = x.size(0)
        
        # If sequence length exceeds max_len, dynamically generate PE (without modifying buffer)
        if seq_len > self.pe.size(0):
            # Generate new PE on-the-fly without modifying the registered buffer
            # This avoids DDP synchronization issues
            pe = self._generate_pe_tensor(seq_len).to(x.device)
        else:
            # Use existing buffer (clone to avoid in-place operations)
            pe = self.pe[:seq_len, :].clone().to(x.device)
        
        x = x + pe
        return self.dropout(x)
    
    def _generate_pe_tensor(self, max_len):
        """Generate positional encoding tensor without registering as buffer"""
        pe = torch.zeros(max_len, self.d_model, device=self.pe.device)
        position = torch.arange(0, max_len, dtype=torch.float, device=pe.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=pe.device).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe


class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding using learnable relative position bias"""
    
    def __init__(self, d_model, dropout=0.1, max_len=512, num_heads=8):
        super(RelativePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len
        self.num_heads = num_heads
        
        # Learnable relative position bias table
        # Shape: (2 * max_len - 1, num_heads)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(2 * max_len - 1, num_heads)
        )
        
        # Initialize relative position bias with very small std for stability
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.005)
        
        # Register buffer for relative position index
        self._generate_relative_position_index(max_len)
    
    def _generate_relative_position_index(self, max_len):
        """Generate relative position index matrix"""
        # Create a matrix where each element (i, j) represents the relative position
        # from position i to position j: j - i
        coords = torch.arange(max_len)
        relative_coords = coords[:, None] - coords[None, :]  # (max_len, max_len)
        relative_coords += max_len - 1  # Shift to make it non-negative
        relative_coords = relative_coords.clamp(0, 2 * max_len - 2)
        self.register_buffer('relative_position_index', relative_coords.long())
    
    def forward(self, x):
        """Forward pass - returns the input unchanged, bias is used in attention"""
        # For relative positional encoding, we don't modify x directly
        # Instead, we return the relative position bias that will be added in attention
        return self.dropout(x)
    
    def get_relative_position_bias(self, seq_len):
        """Get relative position bias for attention computation"""
        if seq_len > self.max_len:
            # Dynamically generate index for longer sequences
            coords = torch.arange(seq_len, device=self.relative_position_index.device)
            relative_coords = coords[:, None] - coords[None, :]
            relative_coords += self.max_len - 1
            relative_coords = relative_coords.clamp(0, 2 * self.max_len - 2)
            relative_position_index = relative_coords.long()
        else:
            relative_position_index = self.relative_position_index[:seq_len, :seq_len]
        
        # Get bias from table: (seq_len, seq_len, num_heads)
        bias = self.relative_position_bias_table[relative_position_index.flatten()].view(
            seq_len, seq_len, self.num_heads
        )
        # Permute to (num_heads, seq_len, seq_len) for attention
        bias = bias.permute(2, 0, 1).contiguous()
        return bias


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, d_model, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        """Forward pass of RMSNorm with numerical stability"""
        # x shape: (seq_len, batch_size, d_model) or (batch_size, d_model)
        # Calculate RMS with numerical stability
        x_squared = x ** 2
        # Clamp to prevent overflow
        x_squared = torch.clamp(x_squared, max=1e6)
        rms = torch.sqrt(torch.mean(x_squared, dim=-1, keepdim=True) + self.eps)
        # Ensure rms is not too small
        rms = torch.clamp(rms, min=self.eps)
        # Normalize
        x_norm = x / rms
        # Clamp normalized values to prevent extreme values
        x_norm = torch.clamp(x_norm, min=-10.0, max=10.0)
        # Scale
        return self.weight * x_norm


class MultiheadAttentionWithRelativeBias(nn.Module):
    """Multi-head attention with relative position bias support"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=False):
        super(MultiheadAttentionWithRelativeBias, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Use standard MultiheadAttention for the base implementation
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=batch_first)
    
    def forward(self, query, key, value, relative_pos_bias=None, attn_mask=None, key_padding_mask=None):
        """
        Forward pass with optional relative position bias
        
        Args:
            query: (seq_len, batch_size, embed_dim) or (batch_size, seq_len, embed_dim) if batch_first
            key: same as query
            value: same as query
            relative_pos_bias: (num_heads, seq_len, seq_len) or None
            attn_mask: (seq_len, seq_len) or None
            key_padding_mask: (batch_size, seq_len) or None
        """
        if relative_pos_bias is None:
            # Use standard attention
            return self.attention(query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        
        # For relative position bias, we need to manually compute attention
        # Get the underlying parameters from the attention module
        if self.batch_first:
            batch_size, seq_len, _ = query.shape
            q = query.transpose(0, 1)  # (seq_len, batch_size, embed_dim)
            k = key.transpose(0, 1)
            v = value.transpose(0, 1)
        else:
            seq_len, batch_size, _ = query.shape
            q, k, v = query, key, value
        
        # Project to get Q, K, V using the attention module's weights
        # We'll use the attention module's in_proj_weight and in_proj_bias
        in_proj_weight = self.attention.in_proj_weight  # (3*embed_dim, embed_dim)
        in_proj_bias = self.attention.in_proj_bias  # (3*embed_dim,)
        
        # Project Q, K, V
        qkv = F.linear(q, in_proj_weight, in_proj_bias)  # (seq_len, batch_size, 3*embed_dim)
        q, k, v = qkv.chunk(3, dim=-1)  # Each: (seq_len, batch_size, embed_dim)
        
        # Reshape for multi-head attention (use contiguous to ensure memory layout)
        q = q.contiguous().view(seq_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)  # (batch_size*num_heads, seq_len, head_dim)
        k = k.contiguous().view(seq_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(seq_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        
        # Compute attention scores
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)  # (batch_size*num_heads, seq_len, seq_len)
        
        # Reshape to add relative position bias
        attn_scores = attn_scores.contiguous().view(batch_size, self.num_heads, seq_len, seq_len)
        
        # Add relative position bias: (num_heads, seq_len, seq_len) -> (batch_size, num_heads, seq_len, seq_len)
        # Scale relative position bias to prevent it from dominating attention scores
        rel_pos_bias_expanded = relative_pos_bias.unsqueeze(0).expand(batch_size, -1, -1, -1)
        # Clamp relative position bias to prevent extreme values
        rel_pos_bias_expanded = torch.clamp(rel_pos_bias_expanded, min=-5.0, max=5.0)
        # Scale relative position bias by a smaller factor to maintain numerical stability
        attn_scores = attn_scores + rel_pos_bias_expanded * 0.05  # Further scale down relative bias
        attn_scores = attn_scores.contiguous().view(batch_size * self.num_heads, seq_len, seq_len)
        
        # Clamp attention scores to prevent overflow in softmax (more conservative)
        attn_scores = torch.clamp(attn_scores, min=-30.0, max=30.0)
        
        # Apply attention mask if provided
        if attn_mask is not None:
            # attn_mask: (seq_len, seq_len)
            attn_mask_expanded = attn_mask.unsqueeze(0).expand(batch_size * self.num_heads, -1, -1)
            attn_scores = attn_scores.masked_fill(attn_mask_expanded == float('-inf'), float('-inf'))
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # key_padding_mask: (batch_size, seq_len) - True/1.0 means mask (set to -inf)
            # Convert to bool if it's float
            if key_padding_mask.dtype == torch.float:
                key_padding_mask = key_padding_mask.bool()
            # Expand to (batch_size, num_heads, 1, seq_len) then reshape
            key_padding_mask_expanded = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            key_padding_mask_expanded = key_padding_mask_expanded.expand(batch_size, self.num_heads, seq_len, -1)
            key_padding_mask_expanded = key_padding_mask_expanded.contiguous().view(batch_size * self.num_heads, seq_len, seq_len)
            attn_scores = attn_scores.masked_fill(key_padding_mask_expanded, float('-inf'))
        
        # Apply softmax with numerical stability
        # Subtract max for numerical stability (doesn't change result due to softmax property)
        attn_scores_max = attn_scores.max(dim=-1, keepdim=True)[0]
        attn_scores_stable = attn_scores - attn_scores_max.detach()
        attn_weights = F.softmax(attn_scores_stable, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Apply attention to values
        attn_output = torch.bmm(attn_weights, v)  # (batch_size*num_heads, seq_len, head_dim)
        
        # Check for NaN before reshaping
        if torch.isnan(attn_output).any():
            attn_output = torch.where(torch.isnan(attn_output), torch.zeros_like(attn_output), attn_output)
        
        # Reshape and project output
        attn_output = attn_output.contiguous().view(batch_size, self.num_heads, seq_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Clamp before linear projection to prevent extreme values
        attn_output = torch.clamp(attn_output, min=-10.0, max=10.0)
        
        attn_output = F.linear(attn_output, self.attention.out_proj.weight, self.attention.out_proj.bias)
        
        # Check for NaN after projection
        if torch.isnan(attn_output).any():
            attn_output = torch.where(torch.isnan(attn_output), torch.zeros_like(attn_output), attn_output)
        
        if self.batch_first:
            return attn_output, attn_weights.view(batch_size, self.num_heads, seq_len, seq_len)
        else:
            return attn_output.transpose(0, 1), attn_weights.view(batch_size, self.num_heads, seq_len, seq_len)


class CustomTransformerEncoderLayer(nn.Module):
    """Custom Transformer Encoder Layer with optional RMSNorm and relative position bias"""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
                 activation=F.relu, norm_type='layernorm', use_relative_pos=False,
                 relative_pos_encoder=None):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.use_relative_pos = use_relative_pos
        if use_relative_pos:
            self.self_attn = MultiheadAttentionWithRelativeBias(d_model, nhead, dropout=dropout, batch_first=False)
        else:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.relative_pos_encoder = relative_pos_encoder
        self.nhead = nhead
        self.d_model = d_model
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Normalization layers
        if norm_type == 'rmsnorm':
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """Forward pass"""
        # Self-attention with residual connection
        src2 = self.norm1(src)
        
        # Get relative position bias if needed
        rel_pos_bias = None
        if self.use_relative_pos and self.relative_pos_encoder is not None:
            seq_len = src2.size(0)
            rel_pos_bias = self.relative_pos_encoder.get_relative_position_bias(seq_len)
        
        # Use attention with or without relative position bias
        if self.use_relative_pos:
            src2, _ = self.self_attn(src2, src2, src2, 
                                     relative_pos_bias=rel_pos_bias,
                                     attn_mask=src_mask, 
                                     key_padding_mask=src_key_padding_mask)
        else:
            src2, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask, 
                                      key_padding_mask=src_key_padding_mask)
        
        # Check for NaN before adding residual
        if torch.isnan(src2).any():
            src2 = torch.where(torch.isnan(src2), torch.zeros_like(src2), src2)
        
        src = src + self.dropout1(src2)
        
        # Check for NaN after residual
        if torch.isnan(src).any():
            src = torch.where(torch.isnan(src), torch.zeros_like(src), src)
        
        # Feedforward with residual connection
        src2 = self.norm2(src)
        
        # Clamp intermediate values to prevent extreme activations
        src2_intermediate = self.linear1(src2)
        src2_intermediate = torch.clamp(src2_intermediate, min=-50.0, max=50.0)
        src2_activated = self.activation(src2_intermediate)
        src2_activated = torch.clamp(src2_activated, min=-10.0, max=10.0)
        src2 = self.linear2(self.dropout(src2_activated))
        
        # Check for NaN before adding residual
        if torch.isnan(src2).any():
            src2 = torch.where(torch.isnan(src2), torch.zeros_like(src2), src2)
        
        src = src + self.dropout2(src2)
        
        # Final NaN check
        if torch.isnan(src).any():
            print(f"Warning: NaN detected in encoder final output, replacing with zeros")
            src = torch.where(torch.isnan(src), torch.zeros_like(src), src)
        
        return src


class CustomTransformerDecoderLayer(nn.Module):
    """Custom Transformer Decoder Layer with optional RMSNorm and relative position bias"""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=F.relu, norm_type='layernorm', use_relative_pos=False,
                 relative_pos_encoder=None):
        super(CustomTransformerDecoderLayer, self).__init__()
        self.use_relative_pos = use_relative_pos
        if use_relative_pos:
            self.self_attn = MultiheadAttentionWithRelativeBias(d_model, nhead, dropout=dropout, batch_first=False)
        else:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.relative_pos_encoder = relative_pos_encoder
        self.nhead = nhead
        self.d_model = d_model
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Normalization layers
        if norm_type == 'rmsnorm':
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
            self.norm3 = RMSNorm(d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """Forward pass"""
        # Self-attention with residual connection
        tgt2 = self.norm1(tgt)
        
        # Get relative position bias if needed
        rel_pos_bias = None
        if self.use_relative_pos and self.relative_pos_encoder is not None:
            seq_len = tgt2.size(0)
            rel_pos_bias = self.relative_pos_encoder.get_relative_position_bias(seq_len)
        
        # Use attention with or without relative position bias
        if self.use_relative_pos:
            tgt2, _ = self.self_attn(tgt2, tgt2, tgt2,
                                     relative_pos_bias=rel_pos_bias,
                                     attn_mask=tgt_mask,
                                     key_padding_mask=tgt_key_padding_mask)
        else:
            tgt2, _ = self.self_attn(tgt2, tgt2, tgt2, attn_mask=tgt_mask,
                                      key_padding_mask=tgt_key_padding_mask)
        
        # Check for NaN
        if torch.isnan(tgt2).any():
            tgt2 = torch.where(torch.isnan(tgt2), torch.zeros_like(tgt2), tgt2)
        
        tgt = tgt + self.dropout1(tgt2)
        
        # Check for NaN after residual
        if torch.isnan(tgt).any():
            tgt = torch.where(torch.isnan(tgt), torch.zeros_like(tgt), tgt)
        
        # Cross-attention with residual connection (no relative position bias for cross-attention)
        tgt2 = self.norm2(tgt)
        tgt2, _ = self.multihead_attn(tgt2, memory, memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)
        
        # Check for NaN
        if torch.isnan(tgt2).any():
            tgt2 = torch.where(torch.isnan(tgt2), torch.zeros_like(tgt2), tgt2)
        
        tgt = tgt + self.dropout2(tgt2)
        
        # Check for NaN after residual
        if torch.isnan(tgt).any():
            tgt = torch.where(torch.isnan(tgt), torch.zeros_like(tgt), tgt)
        
        # Feedforward with residual connection
        tgt2 = self.norm3(tgt)
        
        # Clamp intermediate values to prevent extreme activations
        tgt2_intermediate = self.linear1(tgt2)
        tgt2_intermediate = torch.clamp(tgt2_intermediate, min=-50.0, max=50.0)
        tgt2_activated = self.activation(tgt2_intermediate)
        tgt2_activated = torch.clamp(tgt2_activated, min=-10.0, max=10.0)
        tgt2 = self.linear2(self.dropout(tgt2_activated))
        
        # Check for NaN
        if torch.isnan(tgt2).any():
            tgt2 = torch.where(torch.isnan(tgt2), torch.zeros_like(tgt2), tgt2)
        
        tgt = tgt + self.dropout3(tgt2)
        
        # Final NaN check
        if torch.isnan(tgt).any():
            tgt = torch.where(torch.isnan(tgt), torch.zeros_like(tgt), tgt)
        
        return tgt


class TransformerNMT(nn.Module):
    """Transformer-based Neural Machine Translation Model"""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048,
                 dropout=0.1, max_len=512, pos_encoding_type='absolute', norm_type='layernorm'):
        super(TransformerNMT, self).__init__()
        
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.max_len = max_len
        self.pos_encoding_type = pos_encoding_type
        self.norm_type = norm_type
        
        # Embedding layers
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional encoding
        if pos_encoding_type == 'relative':
            self.pos_encoder = RelativePositionalEncoding(d_model, dropout, max_len, nhead)
            self.use_relative_pos = True
        else:
            self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
            self.use_relative_pos = False
        
        # Transformer layers
        if norm_type == 'rmsnorm' or pos_encoding_type == 'relative':
            # Use custom layers for RMSNorm or relative position encoding
            encoder_layers = [
                CustomTransformerEncoderLayer(
                    d_model, nhead, dim_feedforward, dropout,
                    norm_type=norm_type,
                    use_relative_pos=self.use_relative_pos,
                    relative_pos_encoder=self.pos_encoder if self.use_relative_pos else None
                )
                for _ in range(num_encoder_layers)
            ]
            self.transformer_encoder = nn.ModuleList(encoder_layers)
            
            decoder_layers = [
                CustomTransformerDecoderLayer(
                    d_model, nhead, dim_feedforward, dropout,
                    norm_type=norm_type,
                    use_relative_pos=self.use_relative_pos,
                    relative_pos_encoder=self.pos_encoder if self.use_relative_pos else None
                )
                for _ in range(num_decoder_layers)
            ]
            self.transformer_decoder = nn.ModuleList(decoder_layers)
            self.use_custom_layers = True
        else:
            # Use standard PyTorch layers
            encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
            
            decoder_layers = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            self.transformer_decoder = TransformerDecoder(decoder_layers, num_decoder_layers)
            self.use_custom_layers = False
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize parameters
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.tgt_embedding.weight.data.uniform_(-initrange, initrange)
        self.output_projection.bias.data.zero_()
        self.output_projection.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, 
                src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Forward pass of the transformer model
        
        Args:
            src: source sequence (seq_len, batch_size)
            tgt: target sequence (seq_len, batch_size) 
            src_mask: source attention mask
            tgt_mask: target attention mask
            src_key_padding_mask: source padding mask
            tgt_key_padding_mask: target padding mask
            memory_key_padding_mask: memory padding mask
        """
        # Embedding and positional encoding
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        
        # Transformer encoding
        if self.use_custom_layers:
            memory = src
            for layer in self.transformer_encoder:
                memory = layer(memory, src_mask, src_key_padding_mask)
        else:
            memory = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        
        # Transformer decoding
        if self.use_custom_layers:
            output = tgt
            for layer in self.transformer_decoder:
                output = layer(output, memory, tgt_mask, None,
                             tgt_key_padding_mask, memory_key_padding_mask)
        else:
            output = self.transformer_decoder(tgt, memory, tgt_mask, None,
                                            tgt_key_padding_mask, memory_key_padding_mask)
        
        # Output projection
        output = self.output_projection(output)
        
        return output
    
    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask to prevent attention to future positions"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)  # 生成上三角掩码
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)) # 
        return mask
    
    def create_padding_mask(self, seq, pad_idx):
        """Create padding mask for sequences"""
        return (seq == pad_idx).transpose(0, 1)
    
    def _load_model(self, model_path: str):
        """Load trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        kwargs = params["args"]
        model = TransformerNMT(vocab=params["vocab"], **kwargs)
        model.load_state_dict(params["state_dict"])
        return model


class BeamSearchDecoder:
    """Beam search decoder for inference"""
    
    def __init__(self, model, beam_size=5, max_len=100, sos_idx=1, eos_idx=2, pad_idx=0):
        self.model = model
        self.beam_size = beam_size
        self.max_len = max_len
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
    
    def decode(self, src, src_mask=None, src_key_padding_mask=None):
        """Beam search decoding"""
        device = src.device
        batch_size = src.size(1)
        
        # Encode source
        src_emb = self.model.src_embedding(src) * math.sqrt(self.model.d_model)
        src_emb = self.model.pos_encoder(src_emb)
        memory = self.model.transformer_encoder(src_emb, src_mask, src_key_padding_mask)
        
        # Initialize beam
        beams = [[(torch.tensor([self.sos_idx], device=device), 0.0)] for _ in range(batch_size)]
        
        for step in range(self.max_len):
            new_beams = [[] for _ in range(batch_size)]
            
            for batch_idx in range(batch_size):
                if not beams[batch_idx]:
                    continue
                    
                for seq, score in beams[batch_idx]:
                    if seq[-1].item() == self.eos_idx:
                        new_beams[batch_idx].append((seq, score))
                        continue
                    
                    # Prepare input
                    tgt_input = seq.unsqueeze(1)  # (seq_len, 1)
                    tgt_mask = self.model.generate_square_subsequent_mask(len(seq)).to(device)
                    
                    # Forward pass
                    tgt_emb = self.model.tgt_embedding(tgt_input) * math.sqrt(self.model.d_model)
                    tgt_emb = self.model.pos_encoder(tgt_emb)
                    
                    batch_memory = memory[:, batch_idx:batch_idx+1, :]
                    output = self.model.transformer_decoder(
                        tgt_emb, batch_memory, tgt_mask, None, None, None
                    )
                    
                    # Get next token probabilities
                    next_token_logits = self.model.output_projection(output[-1, 0, :])
                    next_token_probs = F.log_softmax(next_token_logits, dim=-1)
                    
                    # Get top k tokens
                    top_probs, top_indices = torch.topk(next_token_probs, self.beam_size)
                    
                    for prob, idx in zip(top_probs, top_indices):
                        new_seq = torch.cat([seq, idx.unsqueeze(0)])
                        new_score = score + prob.item()
                        new_beams[batch_idx].append((new_seq, new_score))
                
                # Keep top k beams
                new_beams[batch_idx].sort(key=lambda x: x[1], reverse=True)
                new_beams[batch_idx] = new_beams[batch_idx][:self.beam_size]
            
            beams = new_beams
        
        # Return best sequences
        results = []
        for batch_idx in range(batch_size):
            if beams[batch_idx]:
                best_seq, _ = max(beams[batch_idx], key=lambda x: x[1])
                results.append(best_seq)
            else:
                results.append(torch.tensor([self.sos_idx, self.eos_idx], device=device))
        
        return results


def create_model(src_vocab_size, tgt_vocab_size, device, pos_encoding_type='absolute', 
                 norm_type='layernorm', **kwargs):
    """Create and initialize the model
    
    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        device: Device to place model on
        pos_encoding_type: 'absolute' or 'relative' (default: 'absolute')
        norm_type: 'layernorm' or 'rmsnorm' (default: 'layernorm')
        **kwargs: Other model parameters (d_model, nhead, etc.)
    """
    model = TransformerNMT(
        src_vocab_size, tgt_vocab_size, 
        pos_encoding_type=pos_encoding_type,
        norm_type=norm_type,
        **kwargs
    )
    model = model.to(device)
    
    # Initialize with Xavier uniform
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.constant_(p, 0)   # 增加初始化
    
    # Fix LayerNorm initialization for stability
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
    
    # Fix RMSNorm initialization for stability
    for module in model.modules():
        if isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)
    
    return model 