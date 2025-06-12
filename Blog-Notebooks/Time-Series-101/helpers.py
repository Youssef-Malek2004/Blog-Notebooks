import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader


def create_lagged_dataset(series, lags=7):
    X, y = [], []
    for i in range(lags, len(series)):
        X.append(series[i - lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)


def preprocess_series(series, lags=7):
    # Create lagged dataset
    X, y = create_lagged_dataset(series, lags)

    # Normalize
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Train-test split
    split = int(0.8 * len(X_scaled))
    X_train, y_train = X_scaled[:split], y_scaled[:split]
    X_test, y_test = X_scaled[split:], y_scaled[split:]

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train).unsqueeze(-1).float()
    y_train_tensor = torch.tensor(y_train).unsqueeze(-1).float()
    X_test_tensor = torch.tensor(X_test).unsqueeze(-1).float()
    y_test_tensor = torch.tensor(y_test).unsqueeze(-1).float()

    return (
        X_train_tensor,
        y_train_tensor,
        X_test_tensor,
        y_test_tensor,
        scaler_X,
        scaler_y
    )


def get_dataloaders(X_train_tensor, y_train_tensor, batch_size=32):
    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True)


def train_model(model, train_loader, n_epochs=40, lr=0.001, device='cpu'):
    model = model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs} - Loss: {total_loss / len(train_loader):.4f}")

    return model


def evaluate_model(model, X_test_tensor, y_test_tensor, scaler_y, device='cpu'):
    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor.to(device)).cpu().numpy()
        y_true = y_test_tensor.numpy()
        y_pred = scaler_y.inverse_transform(preds)
        y_true = scaler_y.inverse_transform(y_true)
    return y_true, y_pred
