import torch
from torch import nn
import math

from constants import MODEL_DIMENSIONS

def get_policy(name: str = "dqn_simple"):
    if name == "dqn_simple":
        return DQNSimplePolicyNetwork(**MODEL_DIMENSIONS)
    elif name == "dqn_lstm":
        return DQNLSTMPolicyNetwork(**MODEL_DIMENSIONS)
    elif name == "dqn_transformer":
        return DQNTransformerPolicyNetwork(**MODEL_DIMENSIONS)
    elif name == "reinforce_simple":
        return ReinforceSimplePolicyNetwork(**MODEL_DIMENSIONS)
    elif name == "reinforce_lstm":
        return ReinforceLSTMPolicyNetwork(**MODEL_DIMENSIONS)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, device, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).to(device)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(device)
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :]

class DQNSimplePolicyNetwork(nn.Module):
    def __init__(self, **kwargs):
        super(DQNSimplePolicyNetwork, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(kwargs["state_dim"], kwargs["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(kwargs["hidden_dim"], kwargs["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(kwargs["hidden_dim"], kwargs["action_dim"]),
        )

    def reset(self):
        pass

    def forward(self, states):
        """
        Forward pass

        Args:
            states: Tensor of shape [batch_size, seq_len, state_dim].

        Returns:
            action_probs: Tensor of shape [batch_size, seq_len, action_dim].
        """
        q_values = self.fc(states)  # Convert to probabilities
        return q_values
    
class DQNLSTMPolicyNetwork(nn.Module):
    def __init__(self, **kwargs):
        super(DQNLSTMPolicyNetwork, self).__init__()

        self.lstm = nn.LSTM(kwargs["state_dim"], kwargs["hidden_dim"], num_layers=kwargs.lstm_layers, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(kwargs["hidden_dim"], kwargs["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(kwargs["hidden_dim"], kwargs["action_dim"]),
        )

    def reset(self):
        pass

    def forward(self, states):
        """
        Forward pass

        Args:
            states: Tensor of shape [batch_size, seq_len, state_dim].

        Returns:
            action_probs: Tensor of shape [batch_size, seq_len, action_dim].
        """

        hidden_state, _ = self.lstm(states)

        q_values = self.fc(hidden_state[:, -1, :]).unsqueeze(1)
        return q_values
    
class DQNTransformerPolicyNetwork(nn.Module):
    def __init__(self, **kwargs):
        super(DQNTransformerPolicyNetwork, self).__init__()

        self.embedding = nn.Linear(kwargs["state_dim"], kwargs["d_model_transformer"])

        self.positional_encoding = PositionalEncoding(kwargs["d_model_transformer"], kwargs["device"])

        encoder_layer = nn.TransformerEncoderLayer(d_model=kwargs["d_model_transformer"], nhead=kwargs["nhead_transformer"], dim_feedforward=kwargs["dim_feedforward_transformer"], batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=kwargs["n_encoding_layers_transformer"])

        # Output layer to predict q-values
        self.fc_out = nn.Linear(kwargs["d_model_transformer"], kwargs["action_dim"])

    def forward(self, x):
        """
        x: [batch_size, seq_len, state_dim]
        """
        # Embed the input states
        x = self.embedding(x) 

        # Add positional encodings
        x = self.positional_encoding(x)

        # Pass through the transformer encoder
        x = self.transformer_encoder(x)

        # Take the embedding corresponding to the last state in the sequence
        x = x[:, -1, :]

        # Output Q-values
        q_values = self.fc_out(x).unsqueeze(1)
        return q_values

    def reset(self):
        pass


class ReinforceSimplePolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ReinforceSimplePolicyNetwork, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def reset(self):
        pass

    def forward(self, states):
        """
        Forward pass

        Args:
            states: Tensor of shape [batch_size, seq_len, state_dim].

        Returns:
            action_probs: Tensor of shape [batch_size, seq_len, action_dim].
        """
        action_probs = self.fc(states)  # Convert to probabilities
        return action_probs

class ReinforceLSTMPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ReinforceLSTMPolicyNetwork, self).__init__()
        self.lstm = nn.LSTM(state_dim, hidden_dim, batch_first=True)  # LSTM layer
        self.fc = nn.Linear(hidden_dim, action_dim)  # Fully connected layer
        self.softmax = nn.Softmax(dim=-1)  # Action probabilities

        self.hidden_state = None

    def reset(self):
        self.hidden_state = None

    def forward(self, states):
        """
        Forward pass through the LSTM policy network.

        Args:
            states: Tensor of shape [batch_size, seq_len, state_dim].
            hidden: Initial hidden state and cell state (optional).

        Returns:
            action_probs: Tensor of shape [batch_size, seq_len, action_dim].
            hidden: Final hidden state and cell state.
        """
        lstm_out, self.hidden_state = self.lstm(states, self.hidden_state)  # LSTM output
        logits = self.fc(lstm_out)  # Fully connected layer
        action_probs = self.softmax(logits)  # Convert to probabilities
        return action_probs
