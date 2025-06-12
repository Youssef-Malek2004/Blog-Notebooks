import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=512, num_layers=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):  # x shape: (B, S, C)
        out, _ = self.gru(x)  # out: (B, S, H)
        return self.fc(out[:, -1, :])
