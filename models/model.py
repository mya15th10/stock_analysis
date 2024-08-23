from torch import nn
import torch
import torch.nn.functional as F


class cnn_stocks_module(nn.Module):
    OUT_CHANNELS = 15   
    KERNEL_SIZE = 3
    
    def __init__(self, window_length: int):
        super(cnn_stocks_module, self).__init__()

        assert window_length >= self.KERNEL_SIZE
        self.cnn = nn.Conv1d(
            1, # in_channel size
            self.OUT_CHANNELS,
            self.KERNEL_SIZE
        )
        # Thêm Batch Normalization \
        self.batch_norm = nn.BatchNorm1d(num_features=self.OUT_CHANNELS)


        num_scores = window_length - self.KERNEL_SIZE + 1 # number of out values
        self.pool = nn.MaxPool1d(num_scores)
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(self.OUT_CHANNELS, 1, bias=True)

    def forward(self, x):
        out = self.cnn(x.unsqueeze(1))
        out = self.batch_norm(out)
        out = F.relu(out)  # Sử dụng hàm ReLU sau Conv1d để học phi tuyến
        out = self.pool(out).squeeze() 
        out = self.dropout(out)
        out = self.linear(out).squeeze()
        return out
