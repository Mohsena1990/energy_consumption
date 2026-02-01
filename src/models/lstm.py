"""
LSTM model for CO2 forecasting.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler

from .base import BaseForecaster, ModelRegistry
from ..core.logging_utils import get_logger


@ModelRegistry.register('lstm')
class LSTMModel(BaseForecaster):
    """LSTM neural network model for time series forecasting."""

    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'lookback': 4,
            'hidden_size': 32,
            'num_layers': 1,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'batch_size': 16,
            'max_epochs': 200,
            'patience': 20,
            'seed': 42,
            'scaler': 'standard'  # 'standard' or 'robust'
        }
        params = {**default_params, **(params or {})}
        # Validate batch_size - must be at least 1
        if 'batch_size' in params and params['batch_size'] < 1:
            params['batch_size'] = 8  # Use default minimum
        super().__init__('lstm', params)

        # Select scaler based on params (RobustScaler better for outliers)
        scaler_type = params.get('scaler', 'standard')
        if scaler_type == 'robust':
            self.scaler_X = RobustScaler()
            self.scaler_y = RobustScaler()
        else:
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()

        self.device = None
        self.lstm_model = None

    def _check_torch(self):
        """Check if PyTorch is available."""
        try:
            import torch
            import torch.nn as nn
            return True
        except ImportError:
            raise ImportError("PyTorch is required for LSTM. Install with: pip install torch")

    def _create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lookback: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X_seq, y_seq = [], []

        for i in range(lookback, len(X)):
            X_seq.append(X[i-lookback:i])
            y_seq.append(y[i])

        return np.array(X_seq), np.array(y_seq)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LSTMModel':
        logger = get_logger()
        self._check_torch()

        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.debug(f"LSTM using device: {self.device}")

        # Set seed for reproducibility
        torch.manual_seed(self.params['seed'])
        np.random.seed(self.params['seed'])

        # Store feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Scale data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        # Create sequences
        lookback = self.params['lookback']
        X_seq, y_seq = self._create_sequences(X_scaled, y_scaled, lookback)

        if len(X_seq) < 10:
            logger.warning(f"Very few sequences ({len(X_seq)}), LSTM may not train well")

        # Split for validation (last 20%)
        val_size = max(1, int(len(X_seq) * 0.2))
        X_train, X_val = X_seq[:-val_size], X_seq[-val_size:]
        y_train, y_val = y_seq[:-val_size], y_seq[-val_size:]

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)

        # Create data loader
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(self.params['batch_size'], len(X_train)),
            shuffle=True
        )

        # Build model
        input_size = X_scaled.shape[1]
        self.lstm_model = LSTMNetwork(
            input_size=input_size,
            hidden_size=self.params['hidden_size'],
            num_layers=self.params['num_layers'],
            dropout=self.params['dropout']
        ).to(self.device)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.lstm_model.parameters(),
            lr=self.params['learning_rate']
        )

        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(self.params['max_epochs']):
            self.lstm_model.train()
            train_loss = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                output = self.lstm_model(batch_X)
                # Use view(-1) instead of squeeze() to preserve batch dimension correctly
                loss = criterion(output.view(-1), batch_y.view(-1))
                loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.lstm_model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            self.lstm_model.eval()
            with torch.no_grad():
                val_output = self.lstm_model(X_val_t)
                # Use view(-1) instead of squeeze() to preserve batch dimension correctly
                val_loss = criterion(val_output.view(-1), y_val_t.view(-1)).item()

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = self.lstm_model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= self.params['patience']:
                logger.debug(f"Early stopping at epoch {epoch}")
                break

        # Restore best model
        if best_state is not None:
            self.lstm_model.load_state_dict(best_state)

        self.is_fitted = True
        logger.debug(f"LSTM trained for {epoch + 1} epochs, best val loss: {best_val_loss:.6f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        self._check_torch()
        import torch

        if isinstance(X, pd.DataFrame):
            X = X.values

        # Scale features
        X_scaled = self.scaler_X.transform(X)

        # For prediction, we need to handle the lookback requirement
        # If X is shorter than lookback, pad with zeros
        lookback = self.params['lookback']

        if len(X_scaled) < lookback:
            # Pad with last available values
            padding = np.repeat(X_scaled[:1], lookback - len(X_scaled), axis=0)
            X_scaled = np.vstack([padding, X_scaled])

        # Create sequences
        X_seq = []
        for i in range(lookback, len(X_scaled) + 1):
            X_seq.append(X_scaled[i-lookback:i])

        X_seq = np.array(X_seq)

        # Predict
        self.lstm_model.eval()
        X_t = torch.FloatTensor(X_seq).to(self.device)

        with torch.no_grad():
            y_scaled = self.lstm_model(X_t).cpu().numpy()

        # Inverse transform
        y_pred = self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()

        return y_pred

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        # LSTM doesn't have direct feature importance
        # Would need permutation importance or attention mechanism
        return None

    @property
    def supports_shap(self) -> bool:
        return False  # SHAP for LSTM is computationally expensive

    @property
    def interpretability_score(self) -> float:
        return 0.3  # Neural networks are less interpretable

    def __getstate__(self):
        """Custom pickle state for LSTM model."""
        state = self.__dict__.copy()
        # Convert device to string for pickling
        if self.device is not None:
            state['device'] = str(self.device)
        return state

    def __setstate__(self, state):
        """Custom unpickle for LSTM model."""
        self.__dict__.update(state)
        # Restore device
        if self.device is not None and isinstance(self.device, str):
            import torch
            self.device = torch.device(self.device)
            # Move model to device if it exists
            if self.lstm_model is not None:
                self.lstm_model = self.lstm_model.to(self.device)


# Lazy import for PyTorch - defined at module level for pickling support
_torch = None
_nn = None


def _get_torch():
    """Lazy import PyTorch."""
    global _torch, _nn
    if _torch is None:
        import torch
        import torch.nn as nn
        _torch = torch
        _nn = nn
    return _torch, _nn


class LSTMNetwork:
    """
    PyTorch LSTM network wrapper.

    This class dynamically creates an nn.Module subclass at runtime while
    maintaining pickle compatibility by storing the configuration and
    recreating the network on unpickle.
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        """Initialize LSTM network."""
        torch, nn = _get_torch()

        # Store configuration for pickling
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Create the actual PyTorch module
        self._create_network()

    def _create_network(self):
        """Create the underlying PyTorch module."""
        torch, nn = _get_torch()

        # Build the network layers
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        self.dropout_layer = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.hidden_size, 1)

        # Track parameters for optimizer
        self._parameters = list(self.lstm.parameters()) + list(self.fc.parameters())

    def forward(self, x):
        """Forward pass through the network."""
        lstm_out, _ = self.lstm(x)
        # Take the last output
        last_out = lstm_out[:, -1, :]
        out = self.dropout_layer(last_out)
        out = self.fc(out)
        return out

    def __call__(self, x):
        """Make the network callable."""
        return self.forward(x)

    def parameters(self):
        """Return model parameters for optimizer."""
        return self._parameters

    def to(self, device):
        """Move network to device."""
        self.lstm = self.lstm.to(device)
        self.dropout_layer = self.dropout_layer.to(device)
        self.fc = self.fc.to(device)
        self._parameters = list(self.lstm.parameters()) + list(self.fc.parameters())
        return self

    def train(self, mode=True):
        """Set training mode."""
        self.lstm.train(mode)
        self.dropout_layer.train(mode)
        self.fc.train(mode)
        return self

    def eval(self):
        """Set evaluation mode."""
        return self.train(False)

    def state_dict(self):
        """Get state dictionary for saving."""
        return {
            'lstm': self.lstm.state_dict(),
            'fc': self.fc.state_dict()
        }

    def load_state_dict(self, state_dict):
        """Load state dictionary."""
        self.lstm.load_state_dict(state_dict['lstm'])
        self.fc.load_state_dict(state_dict['fc'])

    def __getstate__(self):
        """Custom pickle state - save config and weights."""
        torch, nn = _get_torch()
        state = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'state_dict': self.state_dict()
        }
        return state

    def __setstate__(self, state):
        """Custom unpickle - recreate network from config and weights."""
        self.input_size = state['input_size']
        self.hidden_size = state['hidden_size']
        self.num_layers = state['num_layers']
        self.dropout = state['dropout']
        self._create_network()
        self.load_state_dict(state['state_dict'])


def get_lstm_param_space() -> Dict[str, Any]:
    """Get parameter search space for LSTM optimization."""
    return {
        'lookback': {'type': 'int', 'low': 4, 'high': 12},
        'hidden_size': {'type': 'int', 'low': 16, 'high': 64},
        'num_layers': {'type': 'int', 'low': 1, 'high': 2},
        'dropout': {'type': 'uniform', 'low': 0.1, 'high': 0.5},
        'learning_rate': {'type': 'log_uniform', 'low': 0.0001, 'high': 0.01},
        'batch_size': {'type': 'categorical', 'choices': [8, 16, 32]}
    }
