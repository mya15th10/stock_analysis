import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
data = pd.read_csv(r'C:\IT\stock_lab.uit\stock_analysis\training_set\CP_FPT.csv')  # Thay thế 'CP_FPT.csv' bằng đường dẫn đến tệp CSV của bạn

# 1. Kiểm tra kích thước và thông tin chung
print("Kích thước dữ liệu:", data.shape)  # Số lượng hàng và cột
print("Thông tin chung về dữ liệu:")
print(data.info())  # Thông tin về các cột và kiểu dữ liệu
print("Mô tả dữ liệu:")
print(data.describe())  # Thống kê cơ bản về dữ liệu

# 2. Kiểm tra phân phối của các đặc trưng
for column in data.columns:
    plt.figure(figsize=(10, 4))
    sns.histplot(data[column], kde=True)
    plt.title(f'Phân phối của {column}')
    plt.show()

# 3. Kiểm tra sự biến động theo thời gian (nếu dữ liệu là chuỗi thời gian)
# Giả sử bạn có cột 'Date' là ngày tháng và 'Close' là giá đóng cửa
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])  # Chuyển cột Date thành kiểu datetime
    data.set_index('Date', inplace=True)  # Đặt Date làm chỉ số (index)
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'])  # Thay 'Close' bằng cột bạn muốn kiểm tra
    plt.title('Biến động giá theo thời gian')
    plt.show()

# 4. Kiểm tra tương quan giữa các đặc trưng
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Ma trận tương quan')
plt.show()
