import torch
from torch.utils.data import Dataset  # tạo ra các tập dữ liệu tùy chỉnh để huấn luyện mô hình.

class cnn_stocks_dataset(Dataset):

    #constructor of a class in Python
    def __init__(self, x_tensor: torch.Tensor, y_tensor: torch.Tensor):
        self._x_tensor = x_tensor  #data input 
        self._y_tensor = y_tensor #data output

    #Trả về số lượng mẫu dữ liệu trong tập data
    def __len__(self):
        return self._y_tensor.shape[0]
    
    # Cho phép truy cập dữ liệu mẫu theo chỉ số.
    def __getitem__(self, idx: int):
        return self._x_tensor[idx, :], self._y_tensor[idx]