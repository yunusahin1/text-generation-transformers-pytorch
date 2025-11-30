import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Implements positional encoding for transformer models.
    Adds position information to the input embeddings.
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """
    Decoder-only Transformer model for character-level text generation.
    Similar architecture to GPT but adapted for character-level modeling.
    """
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, 
                 dim_feedforward=1024, dropout=0.1, max_len=5000):
        """
        Args:
            vocab_size: Size of the vocabulary
            d_model: Dimension of the model (embedding dimension)
            nhead: Number of attention heads
            num_layers: Number of transformer decoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            max_len: Maximum sequence length
        """
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection layer
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)
    
    def generate_square_subsequent_mask(self, sz, device):
        """
        Generate a causal mask to prevent attention to future positions.
        
        Args:
            sz: Size of the mask (sequence length)
            device: Device to create the mask on
        
        Returns:
            Mask tensor of shape (sz, sz)
        """
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, src, memory=None):
        """
        Forward pass through the transformer.
        
        Args:
            src: Input tensor of shape (batch_size, seq_len) containing token indices
            memory: Memory tensor from encoder (not used in decoder-only model)
        
        Returns:
            Output logits of shape (batch_size, seq_len, vocab_size)
        """
        device = src.device
        seq_len = src.size(1)
        
        # Create causal mask
        mask = self.generate_square_subsequent_mask(seq_len, device)
        
        # Embedding and positional encoding
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # For decoder-only model, we use the same input as memory
        # This is similar to GPT architecture
        output = self.transformer_decoder(x, x, tgt_mask=mask, memory_mask=mask)
        
        # Project to vocabulary size
        output = self.fc_out(output)
        
        # Return only the last time step for next character prediction
        return output[:, -1, :]
    
    def forward_full_sequence(self, src):
        """
        Forward pass that returns predictions for all positions.
        Useful for training on full sequences.
        
        Args:
            src: Input tensor of shape (batch_size, seq_len) containing token indices
        
        Returns:
            Output logits of shape (batch_size, seq_len, vocab_size)
        """
        device = src.device
        seq_len = src.size(1)
        
        # Create causal mask
        mask = self.generate_square_subsequent_mask(seq_len, device)
        
        # Embedding and positional encoding
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Transformer decoder
        output = self.transformer_decoder(x, x, tgt_mask=mask, memory_mask=mask)
        
        # Project to vocabulary size
        output = self.fc_out(output)
        
        return output
