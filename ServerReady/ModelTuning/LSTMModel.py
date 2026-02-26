import torch
import torch.nn as nn

# -----------------------------
# CONFIG
# -----------------------------
class Config:
    encoder_history = 168      # Number of past timesteps fed to the encoder
    forecast_length = 168      # Number of future timesteps to predict
    encoder_features = 8       # Number of input features for the encoder
    decoder_features = 11      # Number of input features for the decoder
    hidden_size = 128          # LSTM hidden state dimensionality
    num_layers = 2             # Number of stacked LSTM layers
    dropout = 0.2              # Dropout rate for regularization
    epochs = 1             # Total training iterations over the dataset
    batch_size = 16            # Number of samples per training step
    learning_rate = 1e-3       # Optimizer step size

    device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available, else CPU
    output_size = 1            # Number of predicted output variables

# -----------------------------
# MODEL
# https://arxiv.org/abs/1409.3215: ORIGINAL Paper on "Encoder-Decoder"
# https://arxiv.org/pdf/1409.0473: Videreudvikling, handler om en dårlig første prediction can ødelægge resten.
# -----------------------------
class LSTMForecast(nn.Module):
    def __init__(self, config: Config):
        super(LSTMForecast, self).__init__()
        self.config = config

        # ===== ENCODER LSTM ======
        # The encoder's job is to READ and COMPRESS the input sequence (your
        # historical/context window) into a fixed-size summary called the
        # "context vector" or "thought vector".
        #
        # Mechanically, it processes each timestep t=1..T of the input sequence
        # and updates its internal state (h, c) at every step:
        #
        #   h_t, c_t = LSTM_cell(x_t, h_{t-1}, c_{t-1})
        #
        # where:
        #   x_t  → input at timestep t    (shape: [batch, encoder_features])
        #   h_t  → hidden state           (shape: [num_layers, batch, hidden_size])
        #   c_t  → cell state             (shape: [num_layers, batch, hidden_size])
        #
        # After processing the FULL input sequence, the FINAL (h_T, c_T) pair
        # acts as the compressed memory of everything the encoder saw.
        # This is what gets handed off to the decoder.
        self.encoder_lstm = nn.LSTM(
            input_size=config.encoder_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0
        )

        # ====== DECODER LSTM =======
        # The decoder's job is to GENERATE the output sequence (your forecast
        # horizon) step-by-step, conditioned on the encoder's final state.
        #
        # It is initialized with the encoder's (h_T, c_T), meaning it "starts"
        # with the compressed memory of the input sequence already loaded.
        # It then autoregressively produces outputs one timestep at a time:
        #
        #   ŷ_t, (h_t, c_t) = LSTM_cell(d_t, h_{t-1}, c_{t-1})
        #
        # where d_t is either:
        #   → "Teacher forcing" mode (training): the ground-truth target from
        #     the previous step is fed as the next input. Speeds up convergence.
        self.decoder_lstm = nn.LSTM(
            input_size=config.decoder_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0
        )

        # ─── OUTPUT PROJECTION ──────────────────────────────────────────────────
        # Projects the decoder's hidden state at each timestep from hidden_size
        # dimensions down to output_size (e.g. 1 for univariate point forecast,
        # or N for multi-target / quantile forecasting).
        #
        #   ŷ_t = W · h_t + b     where W ∈ ℝ^{output_size × hidden_size}
        self.fc = nn.Linear(config.hidden_size, config.output_size)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            # ===== INPUT-HIDDEN WEIGHTS (weight_ih) — Xavier Uniform =====
            # These weights connect the external input x_t to the four LSTM gates
            # (input, forget, output, gate) at each timestep.
            #
            # Xavier Uniform initialization scales the weights based on the number
            # of incoming and outgoing connections (fan_in, fan_out), keeping the
            # variance of activations consistent across layers. This prevents the
            # signal from shrinking or exploding as it passes forward through the network.
            #
            # Paper: Glorot & Bengio (2010) — "Understanding the difficulty of training
            # deep feedforward neural networks"
            # https://proceedings.mlr.press/v9/glorot10a.html
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            
            # ===== HIDDEN-HIDDEN WEIGHTS (weight_hh) — Orthogonal =====
            # These are the recurrent weights — they connect the hidden state h_{t-1}
            # back into the LSTM cell at the next timestep. This is the weight matrix
            # that gets multiplied repeatedly across every timestep in your sequence.
            #
            # With 168 timesteps, this matrix is applied 168 times in sequence.
            # If its eigenvalues are > 1, gradients explode. If < 1, they vanish.
            # Orthogonal initialization produces a matrix whose eigenvalues all have
            # magnitude exactly 1, keeping the gradient signal stable across the
            # entire sequence length during backpropagation.
            #
            # This is the most impactful initialization in your model given your
            # long sequence length of 168 timesteps.
            #
            # Paper: Saxe, McClelland & Ganguli (2014) — "Exact solutions to the
            # nonlinear dynamics of learning in deep linear networks"
            # https://arxiv.org/abs/1312.6120
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            
            # ===== BIASES — Zero Initialization =====
            # Biases are set to zero as a neutral, clean starting point.
            # There are no strong theoretical reasons to initialize biases
            # differently — zero is standard and works well in practice.
            #
            # One known exception: the forget gate bias in LSTMs is sometimes
            # initialized to 1.0 instead of 0.0, so the network starts by
            # "remembering everything" and learns to forget selectively.
            # This can help with very long sequences but is not always necessary.
            #
            # Paper: Jozefowicz, Zaremba & Sutskever (2015) — "An Empirical
            # Exploration of Recurrent Network Architectures" (see forget gate section)
            # https://proceedings.mlr.press/v37/jozefowicz15.html
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

    def forward(self, encoder_input, decoder_input):
        # Run encoder over historical input; discard outputs, keep final hidden & cell states
        _, (hidden, cell) = self.encoder_lstm(encoder_input)

        # Seed the decoder with encoder's final state — classic seq2seq context passing
        decoder_output, _ = self.decoder_lstm(decoder_input, (hidden, cell))

        # Project decoder hidden states to scalar predictions, remove trailing dim
        output = self.fc(decoder_output).squeeze(-1)
        return output