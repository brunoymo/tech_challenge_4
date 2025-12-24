import yfinance as yf
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# --- 1. Configurações e Coleta de Dados ---
TICKER = 'AAPL'
START_DATE = '2018-01-01'
END_DATE = '2024-07-20'
WINDOW_SIZE = 60 # Janela de 60 dias para prever o próximo

MODEL_PATH = 'api/models/stock_lstm_model.h5'
SCALER_PATH = 'api/models/scaler.pkl'

# Criar diretório de modelos se não existir
os.makedirs('api/models', exist_ok=True)

print(f"Baixando dados para {TICKER} de {START_DATE} até {END_DATE}...")
try:
    df = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
    if df.empty:
        raise ValueError("DataFrame vazio")
    print("Dados baixados com sucesso.")
    print(f"Total de registros: {len(df)}")
except Exception as e:
    print(f"Erro ao baixar dados: {e}")
    print("Criando dados sintéticos para demonstração...")
    # Criar dados sintéticos para teste
    date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='B')  # Business days
    np.random.seed(42)
    # Simular preços de ações com tendência e volatilidade
    base_price = 150
    trend = np.linspace(0, 50, len(date_range))
    volatility = np.random.randn(len(date_range)) * 5
    prices = base_price + trend + volatility
    prices = np.maximum(prices, 1)  # Garantir preços positivos
    
    df = pd.DataFrame({
        'Close': prices,
        'Open': prices * (1 + np.random.randn(len(date_range)) * 0.01),
        'High': prices * (1 + np.abs(np.random.randn(len(date_range))) * 0.02),
        'Low': prices * (1 - np.abs(np.random.randn(len(date_range))) * 0.02),
        'Volume': np.random.randint(1000000, 10000000, len(date_range))
    }, index=date_range)
    print(f"Dados sintéticos criados. Total de registros: {len(df)}")

# --- 2. Pré-processamento dos Dados ---
# Verificar se o DataFrame tem multi-level columns (caso do yfinance atualizado)
if isinstance(df.columns, pd.MultiIndex):
    # Achatar as colunas multi-level
    df.columns = df.columns.get_level_values(0)

data = df.filter(['Close'])
dataset = data.values

# Dividir dados em treino (80%) e teste (20%)
training_data_len = int(len(dataset) * 0.8)

# Escalonamento dos dados
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Criar dados de treino escalonados
train_data = scaled_data[0:training_data_len, :]

# Função para criar sequências
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_data, WINDOW_SIZE)

# Remodelar para o formato 3D [amostras, passos de tempo, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

print(f"Formato dos dados de treino (X_train): {X_train.shape}")

# --- 3. Construção e Treinamento do Modelo LSTM ---
print("Construindo o modelo LSTM...")
model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=32))
model.add(Dense(units=1))

print("Compilando o modelo...")
model.compile(optimizer='adam', loss='mean_squared_error')

print("Iniciando o treinamento do modelo...")
history = model.fit(X_train, y_train, batch_size=32, epochs=50)
print("Treinamento concluído.")

# --- 4. Avaliação do Modelo ---
print("Iniciando avaliação do modelo...")

# Preparar dados de teste
test_data = scaled_data[training_data_len - WINDOW_SIZE:, :]

X_test, y_test_scaled = create_sequences(test_data, WINDOW_SIZE)
y_test_actual = dataset[training_data_len:, :] # y_test real, não escalonado

# Remodelar dados de teste
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Fazer previsões
predictions_scaled = model.predict(X_test)
predictions = scaler.inverse_transform(predictions_scaled)

# Calcular métricas
mae = mean_absolute_error(y_test_actual, predictions)
rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
mape = np.mean(np.abs((y_test_actual - predictions) / y_test_actual)) * 100

print("\n--- Métricas de Avaliação no Conjunto de Teste ---")
print(f'Mean Absolute Error (MAE):    {mae:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')

# --- 5. Salvamento dos Artefatos ---
print(f"\nSalvando modelo em {MODEL_PATH}...")
model.save(MODEL_PATH)

print(f"Salvando escalonador em {SCALER_PATH}...")
with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)

print("Processo concluído. Modelo e escalonador estão prontos para o deploy.")