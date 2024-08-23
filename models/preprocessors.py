import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def process_inputs(price_series: pd.Series, window_length: int) -> pd.DataFrame:
    """
    Tạo ra các chuỗi dữ liệu dựa trên một cửa sổ trượt (moving window).
    Ví dụ, với một cửa sổ dài 10, các chuỗi sẽ bao gồm các chỉ số từ 1-10, 2-11, 3-12, v.v.
    Chuỗi này sẽ được chia tỉ lệ theo giá trị ngay trước giá trị đầu tiên trong cửa sổ trượt.
    :param price_series: Dữ liệu giá cổ phiếu để trích xuất chuỗi.
    :param window_length: Độ dài của cửa sổ trượt.
    T: Số ngày( EX: 10-1=ngày 9, 10-2 = ngày 8) nên phải dùng reversed để đảo
    :return: DataFrame của pandas mà mỗi hàng là một chuỗi dữ liệu, và chỉ số là ngày tham chiếu.
    """

    dataframes = []
    for i in range(window_length):
        dataframes.append(price_series.shift(i). to_frame(f"T-{i}"))

    df = pd.concat(reversed(dataframes), axis=1)

    # Tính phần trăm thay đổi so với giá trị đầu tiên trong mỗi chuỗi con
    df = df.divide(price_series.shift(window_length), axis='rows') - 1

    #Xóa bỏ các hàng có giá trị NaN (không hợp lệ do việc dịch chuyển) và trả về DataFrame kết quả.
    df = df.dropna() 

    # Chuẩn hóa dữ liệu về khoảng [0, 1] bằng MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))
    df[df.columns] = scaler.fit_transform(df[df.columns])

    return df

def process_targets(perf_series: pd.Series) -> pd.Series:
    """
    Tạo ra các mục tiêu (targets) là dữ liệu giá trị sau 2 ngày so với ngày tham chiếu.
    :param perf_series: Dữ liệu lợi nhuận giá cổ phiếu để trích xuất mục tiêu.
    :return: Một chuỗi giá trị của lợi nhuận sau 2 ngày, với chỉ số là ngày tham chiếu.
    """

    return perf_series.pct_change().shift(-2).dropna()
