import torch
import torch.nn as nn

# -----------------------------
# CONFIG
# -----------------------------
class Config:
    # -------------------------
    # Hyperparameters
    # -------------------------
    encoder_history = 168
    forecast_length = 168
    encoder_features = 8
    decoder_features = 11
    hidden_size = 128
    num_layers = 2
    dropout = 0.2
    epochs = 50000             # FIXED: was 'epoch'
    batch_size = 64
    learning_rate = 1e-3

    # -------------------------
    # Device & output
    # -------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_size = 1

# -----------------------------
# MODEL
# -----------------------------
class LSTMForecast(nn.Module):
    def __init__(self, config: Config):
        super(LSTMForecast, self).__init__()
        self.config = config

        self.encoder_lstm = nn.LSTM(
            input_size=config.encoder_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0
        )

        self.decoder_lstm = nn.LSTM(
            input_size=config.decoder_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(config.hidden_size, config.output_size)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

    def forward(self, encoder_input, decoder_input):
        _, (hidden, cell) = self.encoder_lstm(encoder_input)
        decoder_output, _ = self.decoder_lstm(decoder_input, (hidden, cell))
        output = self.fc(decoder_output).squeeze(-1)
        return output