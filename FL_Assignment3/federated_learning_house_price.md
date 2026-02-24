# Federated Learning for House Price Prediction

## Complete Implementation with FedAvg Algorithm

This notebook demonstrates a **Federated Learning (FL)** framework where multiple clients collaboratively train a linear regression model for house price prediction without sharing their raw data. The central server uses **Federated Averaging (FedAvg)** to aggregate local model parameters into a global model.[web:28][web:34]

---

## Cell 1: Import Required Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully!")
```

---

## Cell 2: Load and Explore the Housing Dataset

```python
# Load the housing price dataset
df = pd.read_csv('housing_price_dataset.csv')

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())
```

---

## Cell 3: Data Preprocessing - Handling Missing Values & Normalization

```python
# Handle missing values (if any) by filling with mean
df_clean = df.copy()
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    if df_clean[col].isnull().sum() > 0:
        mean_value = df_clean[col].mean()
        df_clean[col].fillna(mean_value, inplace=True)

print("Missing values after imputation:")
print(df_clean.isnull().sum())

# Separate features and target
X = df_clean.drop('price', axis=1)
y = df_clean['price']

# Normalize features using StandardScaler
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
X_normalized = pd.DataFrame(X_normalized, columns=X.columns)

print("\nFeatures normalized successfully!")
print("Feature columns:", list(X.columns))
```

---

## Cell 4: Split Data Among Multiple Clients (Federated Data Distribution)

```python
# Configuration
NUM_CLIENTS = 5  # Number of participating devices/nodes
TEST_SIZE = 0.2  # Holdout test set at server

# Split data into train and test (test remains at server for evaluation)
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y, test_size=TEST_SIZE, random_state=42
)

print(f"Total training samples: {len(X_train)}")
print(f"Total test samples: {len(X_test)}")

# Shuffle and split training data among clients
np.random.seed(42)
indices = np.random.permutation(len(X_train))
client_indices = np.array_split(indices, NUM_CLIENTS)

# Create client datasets (each client gets its own private data)
client_datasets = []
for i, idx in enumerate(client_indices):
    client_X = X_train.iloc[idx].reset_index(drop=True)
    client_y = y_train.iloc[idx].reset_index(drop=True)
    client_datasets.append((client_X, client_y))
    print(f"Client {i+1}: {len(client_X)} samples")

# Save client datasets to separate CSV files (simulating distributed storage)
CLIENT_DATA_DIR = "federated_clients_data"
os.makedirs(CLIENT_DATA_DIR, exist_ok=True)

for i, (client_X, client_y) in enumerate(client_datasets, start=1):
    client_df = client_X.copy()
    client_df['price'] = client_y
    client_path = os.path.join(CLIENT_DATA_DIR, f"client_{i}_data.csv")
    client_df.to_csv(client_path, index=False)
    print(f"Saved: {client_path}")
```

---

## Cell 5: Define Linear Regression Model Class

```python
class LinearRegressionModel:
    """Simple Linear Regression model for federated learning"""
    
    def __init__(self, n_features):
        self.n_features = n_features
        # Initialize weights and bias with small random values
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
    
    def predict(self, X):
        """Make predictions"""
        return np.dot(X, self.weights) + self.bias
    
    def compute_loss(self, X, y):
        """Compute Mean Squared Error loss"""
        predictions = self.predict(X)
        return np.mean((predictions - y) ** 2)
    
    def get_parameters(self):
        """Get model parameters"""
        return {'weights': self.weights.copy(), 'bias': self.bias}
    
    def set_parameters(self, params):
        """Set model parameters"""
        self.weights = params['weights'].copy()
        self.bias = params['bias']
    
    def train_step(self, X, y, learning_rate=0.01):
        """Single training step using gradient descent"""
        n_samples = len(X)
        
        # Forward pass
        predictions = self.predict(X)
        
        # Compute gradients
        error = predictions - y
        weight_gradient = (2 / n_samples) * np.dot(X.T, error)
        bias_gradient = (2 / n_samples) * np.sum(error)
        
        # Update parameters
        self.weights -= learning_rate * weight_gradient
        self.bias -= learning_rate * bias_gradient
        
        return self.compute_loss(X, y)

print("Linear Regression Model class defined!")
```

---

## Cell 6: Client Training Function

```python
def train_client_model(client_X, client_y, global_params, local_epochs=5, learning_rate=0.01):
    """
    Train a local model on client's private data
    
    Args:
        client_X: Client's feature data
        client_y: Client's target data
        global_params: Global model parameters from server
        local_epochs: Number of local training epochs
        learning_rate: Learning rate for gradient descent
    
    Returns:
        Updated local model parameters and training loss
    """
    n_features = client_X.shape[1]
    
    # Initialize local model with global parameters
    local_model = LinearRegressionModel(n_features)
    local_model.set_parameters(global_params)
    
    # Train locally for specified epochs
    losses = []
    for epoch in range(local_epochs):
        loss = local_model.train_step(
            client_X.values, 
            client_y.values, 
            learning_rate
        )
        losses.append(loss)
    
    # Return updated parameters (NOT raw data - preserving privacy!)
    return local_model.get_parameters(), losses[-1], len(client_X)

print("Client training function defined!")
```

---

## Cell 7: Federated Averaging (FedAvg) Function

```python
def federated_averaging(client_params_list, client_weights):
    """
    Aggregate client model parameters using weighted averaging (FedAvg algorithm)
    
    Args:
        client_params_list: List of parameter dictionaries from all clients
        client_weights: List of weights for each client (typically number of samples)
    
    Returns:
        Aggregated global model parameters
    """
    total_weight = sum(client_weights)
    
    # Initialize aggregated parameters
    aggregated_params = {
        'weights': np.zeros_like(client_params_list[0]['weights']),
        'bias': 0.0
    }
    
    # Weighted average of all client parameters
    for params, weight in zip(client_params_list, client_weights):
        aggregated_params['weights'] += (weight / total_weight) * params['weights']
        aggregated_params['bias'] += (weight / total_weight) * params['bias']
    
    return aggregated_params

print("Federated Averaging (FedAvg) function defined!")
```

---

## Cell 8: Server Evaluation Function

```python
def evaluate_global_model(global_params, X_test, y_test):
    """
    Evaluate global model on test data at server
    
    Args:
        global_params: Global model parameters
        X_test: Test features
        y_test: Test targets
    
    Returns:
        Dictionary with evaluation metrics
    """
    n_features = X_test.shape[1]
    model = LinearRegressionModel(n_features)
    model.set_parameters(global_params)
    
    # Make predictions
    predictions = model.predict(X_test.values)
    
    # Compute metrics
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2
    }

print("Evaluation function defined!")
```

---

## Cell 9: Run Federated Learning Training Loop

```python
# Federated Learning Configuration
NUM_ROUNDS = 20  # Number of federated learning rounds
LOCAL_EPOCHS = 5  # Number of epochs each client trains locally
LEARNING_RATE = 0.01

# Initialize global model
n_features = X_train.shape[1]
global_model = LinearRegressionModel(n_features)
global_params = global_model.get_parameters()

# Storage for tracking progress
history = {
    'round': [],
    'train_loss': [],
    'test_mse': [],
    'test_rmse': [],
    'test_mae': [],
    'test_r2': []
}

print("=" * 60)
print("FEDERATED LEARNING TRAINING STARTED")
print("=" * 60)
print(f"Number of Clients: {NUM_CLIENTS}")
print(f"Number of Rounds: {NUM_ROUNDS}")
print(f"Local Epochs per Round: {LOCAL_EPOCHS}")
print(f"Learning Rate: {LEARNING_RATE}")
print("=" * 60)

# Federated Learning Training Loop
for round_num in range(1, NUM_ROUNDS + 1):
    print(f"\n{'='*60}")
    print(f"ROUND {round_num}/{NUM_ROUNDS}")
    print(f"{'='*60}")
    
    # Step 1: Each client trains locally on its private data
    client_params_list = []
    client_weights_list = []
    round_losses = []
    
    for client_id, (client_X, client_y) in enumerate(client_datasets, start=1):
        # Client trains locally without sharing raw data
        updated_params, loss, n_samples = train_client_model(
            client_X, 
            client_y, 
            global_params,
            local_epochs=LOCAL_EPOCHS,
            learning_rate=LEARNING_RATE
        )
        
        client_params_list.append(updated_params)
        client_weights_list.append(n_samples)
        round_losses.append(loss)
        
        print(f"  Client {client_id}: Loss = {loss:.2f}, Samples = {n_samples}")
    
    # Step 2: Server aggregates client parameters using FedAvg
    global_params = federated_averaging(client_params_list, client_weights_list)
    avg_train_loss = np.mean(round_losses)
    
    # Step 3: Evaluate global model on server's test set
    metrics = evaluate_global_model(global_params, X_test, y_test)
    
    # Store results
    history['round'].append(round_num)
    history['train_loss'].append(avg_train_loss)
    history['test_mse'].append(metrics['mse'])
    history['test_rmse'].append(metrics['rmse'])
    history['test_mae'].append(metrics['mae'])
    history['test_r2'].append(metrics['r2_score'])
    
    print(f"\n  → Average Train Loss: {avg_train_loss:.2f}")
    print(f"  → Test RMSE: {metrics['rmse']:.2f}")
    print(f"  → Test MAE: {metrics['mae']:.2f}")
    print(f"  → Test R² Score: {metrics['r2_score']:.4f}")

print("\n" + "=" * 60)
print("FEDERATED LEARNING TRAINING COMPLETED!")
print("=" * 60)
```

---

## Cell 10: Visualize Training Progress

```python
# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Training Loss
axes[0, 0].plot(history['round'], history['train_loss'], 
                marker='o', linewidth=2, color='#2563eb')
axes[0, 0].set_xlabel('Federated Round', fontsize=12)
axes[0, 0].set_ylabel('Average Training Loss (MSE)', fontsize=12)
axes[0, 0].set_title('Training Loss over Rounds', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Test RMSE
axes[0, 1].plot(history['round'], history['test_rmse'], 
                marker='s', linewidth=2, color='#dc2626')
axes[0, 1].set_xlabel('Federated Round', fontsize=12)
axes[0, 1].set_ylabel('Test RMSE', fontsize=12)
axes[0, 1].set_title('Test RMSE over Rounds', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Test MAE
axes[1, 0].plot(history['round'], history['test_mae'], 
                marker='^', linewidth=2, color='#16a34a')
axes[1, 0].set_xlabel('Federated Round', fontsize=12)
axes[1, 0].set_ylabel('Test MAE', fontsize=12)
axes[1, 0].set_title('Test MAE over Rounds', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: R² Score
axes[1, 1].plot(history['round'], history['test_r2'], 
                marker='d', linewidth=2, color='#9333ea')
axes[1, 1].set_xlabel('Federated Round', fontsize=12)
axes[1, 1].set_ylabel('R² Score', fontsize=12)
axes[1, 1].set_title('R² Score over Rounds', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('federated_learning_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualization saved as 'federated_learning_results.png'")
```

---

## Cell 11: Final Model Evaluation and Summary

```python
# Final evaluation
final_metrics = evaluate_global_model(global_params, X_test, y_test)

print("\n" + "=" * 60)
print("FINAL GLOBAL MODEL PERFORMANCE")
print("=" * 60)
print(f"Mean Squared Error (MSE):  {final_metrics['mse']:.2f}")
print(f"Root Mean Squared Error (RMSE): {final_metrics['rmse']:.2f}")
print(f"Mean Absolute Error (MAE): {final_metrics['mae']:.2f}")
print(f"R² Score: {final_metrics['r2_score']:.4f}")
print("=" * 60)

# Display model parameters
print("\nGlobal Model Parameters:")
print(f"Weights: {global_params['weights']}")
print(f"Bias: {global_params['bias']:.2f}")

# Sample predictions
print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS (First 10 test samples)")
print("=" * 60)

global_model.set_parameters(global_params)
sample_predictions = global_model.predict(X_test.values[:10])

comparison_df = pd.DataFrame({
    'Actual Price': y_test.values[:10],
    'Predicted Price': sample_predictions,
    'Absolute Error': np.abs(y_test.values[:10] - sample_predictions)
})

print(comparison_df.to_string(index=True))
print("=" * 60)
```

---

## Cell 12: Save Training History

```python
# Save training history to CSV
history_df = pd.DataFrame(history)
history_df.to_csv('federated_learning_history.csv', index=False)
print("Training history saved to 'federated_learning_history.csv'")

# Display summary
print("\nTraining History Summary:")
print(history_df.to_string(index=False))
```

---

## Summary

This notebook demonstrates a complete **Federated Learning framework** using **FedAvg algorithm** for house price prediction:

### Key Features:
1. **Data Privacy**: Each client trains on local data; only model parameters are shared
2. **Federated Averaging (FedAvg)**: Server aggregates client parameters using weighted averaging[web:28][web:34]
3. **Linear Regression**: Simple yet effective model for regression tasks
4. **Multiple Clients**: Simulates distributed learning across 5 clients
5. **Evaluation Metrics**: MSE, RMSE, MAE, and R² score tracking

### Privacy Preservation:
- ✅ Raw data never leaves client devices
- ✅ Only model parameters (weights, bias) are communicated
- ✅ Server aggregates parameters without accessing private data
- ✅ Collaborative learning without centralized data collection

### Dataset Information:
- **Features**: `avg_area_income`, `avg_area_house_age`, `avg_area_rooms`, `avg_area_bedrooms`, `area_population`
- **Target**: `price`
- **Total Samples**: 1000 houses
- **Training Samples**: 800 (distributed among 5 clients)
- **Test Samples**: 200 (held at server for evaluation)

---

## References
- McMahan et al. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data" - Original FedAvg paper[web:28]
- Flower Framework FedAvg Documentation[web:34]
- Kaggle Housing Datasets[web:32][web:35][web:37]
