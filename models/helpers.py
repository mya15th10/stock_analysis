import logging
import torch
from torch.utils.data import DataLoader
from functools import partial
import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt
from scipy.stats import norm
from models.model import cnn_stocks_module
from models.dataset import cnn_stocks_dataset


LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-6

# Thiết lập logging
def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    return logger

# Sử dụng logging để theo dõi quá trình huấn luyện
logger = setup_logging()

def draw_chart(y_series: pd.Series):
    y_series.plot.hist(bins=50, label='Target Returns')

    mu, stdev = norm.fit(y_series)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 50)
    p = norm.pdf(x, mu, stdev)
    p *= y_series.shape[0] / p.sum()

    plt.plot(x, p, 'r', linewidth=2, label='Gaussian Curve')
    plt.legend()

def chart_y_histogram(y_series: pd.Series):
    draw_chart(y_series)
    plt.show()

    draw_chart(y_series)
    xmin, _ = plt.xlim()
    plt.axis([xmin, -0.015, 0, 10])
    plt.show()

def train(x_df: pd.DataFrame, y_series: pd.Series, epochs: int = 100, patience: int = 10, model_save_path: str = 'best_model.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_tensor, y_tensor = torch.tensor(x_df.values).float(), torch.tensor(y_series.values).float()
    x_tensor, y_tensor = x_tensor.to(device), y_tensor.to(device)

    train_dataset = cnn_stocks_dataset(x_tensor, y_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = cnn_stocks_module(x_df.shape[1]).to(device).train()
    # Khởi tạo optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # Khởi tạo Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    loss_func = partial(torch.nn.functional.huber_loss, delta=0.5)

    best_loss = float('inf')
    patience_counter = 0
    loss_values = []

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0

        for x_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            out = model(x_batch)
            loss = loss_func(out, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        elapsed_time = time.time() - start_time
        loss_values.append(avg_loss)
        scheduler.step(avg_loss)


        # Early Stopping Logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0  # Reset lại bộ đếm nếu có cải thiện
            torch.save(model.state_dict(), model_save_path)  # Lưu mô hình tốt nhất
            logger.info(f"Saved better model at epoch {epoch+1} with loss {avg_loss:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

        logger.info(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Time: {elapsed_time:.2f}s')

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Saved better model at epoch {epoch+1} with loss {avg_loss:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    plt.plot(range(1, len(loss_values) + 1), loss_values, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return model

def predict(trained_model, x_df: pd.DataFrame) -> pd.Series:
    trained_model.eval()

    x_tensor = torch.tensor(x_df.values).float().to(next(trained_model.parameters()).device)
    prediction = trained_model(x_tensor)

    return pd.Series(prediction.cpu().detach().numpy(), index=x_df.index)
