"""
Keras model classes for the Seq2Seq model.

This module defines the custom Keras model classes needed to load and use
the pre-trained Seq2Seq model with attention mechanism.

Models and components:
- Encoder: LSTM-based encoder that processes input sequences
- BahdanauAttention: Attention mechanism that focuses on relevant parts of encoder output
- Decoder: LSTM-based decoder with attention that generates output sequences
- Seq2Seq: Full sequence-to-sequence model that combines encoder and decoder

These classes are designed to match the exact architecture used during training
to ensure proper model loading and inference.
"""

import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class Encoder(tf.keras.Model):
    """
    Encoder component of the Seq2Seq model.
    
    The encoder converts an input sequence into a fixed-size representation
    and returns both the sequence of hidden states and the final state.
    
    Attributes:
        vocab_size (int): Size of the vocabulary
        embedding_dim (int): Dimension of the embedding vectors
        enc_units (int): Number of LSTM units
        embedding: Embedding layer
        lstm: LSTM layer with return_sequences and return_state
    """
    
    def __init__(self, vocab_size=10001, embedding_dim=128, enc_units=256, **kwargs):
        """
        Initialize the encoder.
        
        Args:
            vocab_size (int): Size of the vocabulary
            embedding_dim (int): Dimension of the embedding vectors
            enc_units (int): Number of LSTM units
            **kwargs: Additional arguments to pass to the Model constructor
        """
        super(Encoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.lstm = tf.keras.layers.LSTM(enc_units, return_sequences=True, return_state=True)

    def call(self, x, training=False):
        """
        Forward pass of the encoder.
        
        Args:
            x: Input tensor of token indices
            training (bool): Whether in training mode
            
        Returns:
            tuple: (outputs, state_h, state_c)
                - outputs: Sequence of hidden states
                - state_h: Final hidden state
                - state_c: Final cell state
        """
        x = self.embedding(x)
        outputs, state_h, state_c = self.lstm(x, training=training)
        return outputs, state_h, state_c

    def get_config(self):
        """
        Get configuration for serialization.
        
        Returns:
            dict: Configuration dictionary
        """
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "enc_units": self.enc_units
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Create an instance from configuration.
        
        Args:
            config (dict): Configuration dictionary
            
        Returns:
            Encoder: New instance
        """
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class BahdanauAttention(tf.keras.layers.Layer):
    """
    Bahdanau attention mechanism.
    
    This attention mechanism computes context vectors by weighting encoder outputs
    based on their relevance to the current decoder state.
    
    Attributes:
        units (int): Dimension of the attention weights
        W1: Dense layer for transforming encoder outputs
        W2: Dense layer for transforming decoder hidden state
        V: Dense layer for computing attention scores
    """
    
    def __init__(self, units=256, **kwargs):
        """
        Initialize the attention mechanism.
        
        Args:
            units (int): Dimension of the attention weights
            **kwargs: Additional arguments to pass to the Layer constructor
        """
        super(BahdanauAttention, self).__init__(**kwargs)
        self.units = units
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, decoder_hidden, encoder_outputs):
        """
        Compute attention weights and context vector.
        
        Args:
            decoder_hidden: Current decoder hidden state
            encoder_outputs: Sequence of encoder hidden states
            
        Returns:
            tuple: (context_vector, attention_weights)
                - context_vector: Weighted sum of encoder outputs
                - attention_weights: Attention scores for each encoder output
        """
        decoder_hidden_time_axis = tf.expand_dims(decoder_hidden, 1)
        score = self.V(tf.nn.tanh(self.W1(encoder_outputs) + self.W2(decoder_hidden_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * encoder_outputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

    def get_config(self):
        """
        Get configuration for serialization.
        
        Returns:
            dict: Configuration dictionary
        """
        config = super().get_config()
        config.update({"units": self.units})
        return config

    @classmethod
    def from_config(cls, config):
        """
        Create an instance from configuration.
        
        Args:
            config (dict): Configuration dictionary
            
        Returns:
            BahdanauAttention: New instance
        """
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class Decoder(tf.keras.Model):
    """
    Decoder component of the Seq2Seq model with attention.
    
    The decoder generates output sequences one token at a time,
    using attention to focus on relevant parts of the encoder output.
    
    Attributes:
        vocab_size (int): Size of the vocabulary
        embedding_dim (int): Dimension of the embedding vectors
        dec_units (int): Number of LSTM units
        embedding: Embedding layer
        lstm: LSTM layer with return_sequences and return_state
        attention: Attention mechanism
        fc: Dense layer for output projection
    """
    
    def __init__(self, vocab_size=10001, embedding_dim=128, dec_units=256, **kwargs):
        """
        Initialize the decoder.
        
        Args:
            vocab_size (int): Size of the vocabulary
            embedding_dim (int): Dimension of the embedding vectors
            dec_units (int): Number of LSTM units
            **kwargs: Additional arguments to pass to the Model constructor
        """
        super(Decoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.lstm = tf.keras.layers.LSTM(dec_units, return_sequences=True, return_state=True)
        self.attention = BahdanauAttention(dec_units)
        self.fc = tf.keras.layers.Dense(vocab_size, activation='softmax')

    def call(self, x, initial_state, encoder_outputs, training=False):
        """
        Forward pass of the decoder.
        
        Args:
            x: Input tensor of token indices
            initial_state: Initial LSTM state (hidden and cell states)
            encoder_outputs: Sequence of encoder hidden states
            training (bool): Whether in training mode
            
        Returns:
            During training:
                output: Logits for next token prediction
            During inference:
                tuple: (output, state_h, state_c, attention_weights)
                    - output: Logits for next token prediction
                    - state_h: Updated hidden state
                    - state_c: Updated cell state
                    - attention_weights: Attention weights for each encoder output
        """
        x = self.embedding(x)
        context_vector, attention_weights = self.attention(initial_state[0], encoder_outputs)
        context_vector = tf.expand_dims(context_vector, 1)
        context_vector = tf.tile(context_vector, [1, tf.shape(x)[1], 1])
        
        x = tf.concat([context_vector, x], axis=-1)
        output, state_h, state_c = self.lstm(x, initial_state=initial_state, training=training)
        output = self.fc(output)
        
        if training:
            return output
        else:
            return output, state_h, state_c, attention_weights

    def get_config(self):
        """
        Get configuration for serialization.
        
        Returns:
            dict: Configuration dictionary
        """
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "dec_units": self.dec_units
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Create an instance from configuration.
        
        Args:
            config (dict): Configuration dictionary
            
        Returns:
            Decoder: New instance
        """
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class Seq2Seq(tf.keras.Model):
    """
    Full sequence-to-sequence model with attention.
    
    This model combines the encoder and decoder components into a complete
    sequence-to-sequence model for text correction.
    
    Attributes:
        encoder: Encoder component
        decoder: Decoder component with attention
    """
    
    def __init__(self, encoder, decoder, **kwargs):
        """
        Initialize the Seq2Seq model.
        
        Args:
            encoder: Encoder component
            decoder: Decoder component
            **kwargs: Additional arguments to pass to the Model constructor
        """
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs, training=False):
        """
        Forward pass of the Seq2Seq model.
        
        Args:
            inputs: Tuple of (encoder_input, decoder_input)
            training (bool): Whether in training mode
            
        Returns:
            During training:
                decoder_outputs: Logits for next token prediction
        """
        encoder_input, decoder_input = inputs
        
        encoder_outputs, state_h, state_c = self.encoder(encoder_input)
        decoder_outputs, _, _, _ = self.decoder(
            decoder_input, [state_h, state_c], encoder_outputs, training=training
        )
        return decoder_outputs
    
    def get_config(self):
        """
        Get configuration for serialization.
        
        Returns:
            dict: Configuration dictionary
        """
        config = super().get_config()
        config.update({
            "encoder": tf.keras.saving.serialize_keras_object(self.encoder),
            "decoder": tf.keras.saving.serialize_keras_object(self.decoder),
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Create an instance from configuration.
        
        Args:
            config (dict): Configuration dictionary
            
        Returns:
            Seq2Seq: New instance
        """
        encoder = tf.keras.saving.deserialize_keras_object(config["encoder"])
        decoder = tf.keras.saving.deserialize_keras_object(config["decoder"])
        return cls(encoder=encoder, decoder=decoder) 