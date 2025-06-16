import torch

INT_2_DIR = {
    0: "left",
    1: "right",
    2: "straight",
    3: "back"
}

MODEL_DIMENSIONS = {
    "state_dim": 11,
    "action_dim": 4,
    "hidden_dim": 128,
    "lstm_layers": 1,
    "d_model_transformer": 128,
    "nhead_transformer": 4,
    "dim_feedforward_transformer": 512,
    "n_encoding_layers_transformer": 2,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
}