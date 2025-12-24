from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
from typing import List, Optional
import numpy as np
import pickle
from keras.models import load_model
import os
import time
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# --- 1. Configuração e Inicialização ---
app = FastAPI(
    title="API de Previsão de Ações com LSTM",
    description="Uma API para prever o preço de fechamento de ações usando um modelo LSTM treinado.",
    version="1.0.0"
)

# Definir o tamanho da janela (deve ser o mesmo usado no treino)
WINDOW_SIZE = 60

# --- 2. Carregamento dos Modelos ---
# Usar caminho absoluto baseado na localização deste arquivo (main.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'stock_lstm_model.h5')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

model = None
scaler = None

@app.on_event("startup")
async def load_artifacts():
    """
    Carrega o modelo e o escalonador na memória quando a aplicação inicia.
    """
    global model, scaler
    print("=" * 50)
    print("Iniciando carregamento de artefatos...")
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"MODEL_PATH: {MODEL_PATH}")
    print(f"SCALER_PATH: {SCALER_PATH}")
    print(f"Modelo existe? {os.path.exists(MODEL_PATH)}")
    print(f"Scaler existe? {os.path.exists(SCALER_PATH)}")
    
    # Listar arquivos no diretório api/models
    models_dir = os.path.join(BASE_DIR, 'models')
    if os.path.exists(models_dir):
        print(f"Arquivos em {models_dir}:")
        for f in os.listdir(models_dir):
            print(f"  - {f}")
    else:
        print(f"Diretório {models_dir} não existe!")
    print("=" * 50)
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Erro: Arquivos de modelo ou escalonador não encontrados.")
        print("Certifique-se de executar o script 'train_model.py' primeiro.")
        return

    try:
        model = load_model(MODEL_PATH)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print("✅ Artefatos carregados com sucesso!")
    except Exception as e:
        print(f"❌ Erro crítico ao carregar artefatos: {e}")

# Middleware para loggar tempo de resposta (Monitoramento)
@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    print(f"Requisição {request.url.path} levou {process_time:.4f}s")
    return response

# --- 3. Definição dos Schemas de Dados (Pydantic) ---
class StockHistory(BaseModel):
    """
    Schema de entrada esperado pela API.
    Espera uma lista de floats com exatamente WINDOW_SIZE elementos.
    """
    historical_prices: List[float]

    class Config:
        json_schema_extra = {
            "example": {
                "historical_prices": [150.1, 151.2, 150.5, 152.0, 153.1] + [160.0] * 55
            }
        }

class PredictionResponse(BaseModel):
    """
    Schema de saída da API.
    """
    predicted_next_day_close_price: float

# --- 4. Endpoints da API ---
@app.get("/", tags=["Health Check"])
def read_root():
    """
    Endpoint de verificação de saúde.
    """
    if model is None or scaler is None:
        return {"status": "AVISO: API está no ar, mas os artefatos do modelo não foram carregados."}
    return {"status": "API está funcionando e o modelo está carregado."}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_stock_price(stock_data: StockHistory):
    """
    Recebe os últimos 60 dias de preços de fechamento e prevê o preço do próximo dia.
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503,
                            detail="Modelo ou escalonador não estão carregados. Verifique os logs do servidor.")

    input_data = stock_data.historical_prices

    # 1. Validar o tamanho da entrada
    if len(input_data)!= WINDOW_SIZE:
        raise HTTPException(status_code=400,
                            detail=f"A entrada deve conter exatamente {WINDOW_SIZE} preços históricos.")

    try:
        # 2. Pré-processamento (Converter, Escalonar, Remodelar)
        input_array = np.array(input_data).reshape(-1, 1)
        scaled_input = scaler.transform(input_array)
        reshaped_input = np.reshape(scaled_input, (1, WINDOW_SIZE, 1))

        # 3. Fazer a previsão
        prediction_scaled = model.predict(reshaped_input)

        # 4. Desfazer o escalonamento
        prediction = scaler.inverse_transform(prediction_scaled)

        # 5. Retornar o resultado
        return {"predicted_next_day_close_price": float(prediction)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno durante a previsão: {str(e)}")


@app.get("/predict-auto/{codigo_acao}", tags=["Prediction"])
def predict_stock_auto(codigo_acao: str):
    """
    Busca automaticamente os últimos 60 dias de preços do código da ação e faz a previsão.
    Exemplo: /predict-auto/AAPL
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503,
                            detail="Modelo ou escalonador não estão carregados. Verifique os logs do servidor.")

    try:
        # 1. Baixar os últimos 60+ dias de dados (pegar um pouco mais para garantir 60 dias úteis)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)  # 90 dias para garantir 60 dias úteis
        
        df = yf.download(codigo_acao.upper(), start=start_date.strftime('%Y-%m-%d'), 
                        end=end_date.strftime('%Y-%m-%d'), progress=False)
        
        if df.empty:
            raise HTTPException(status_code=404, 
                              detail=f"Não foi possível obter dados para o código {codigo_acao}. Verifique se o símbolo está correto.")
        
        # 2. Processar o DataFrame (lidar com multi-level columns se necessário)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # 3. Pegar apenas a coluna Close
        close_prices = df['Close'].values
        
        # 4. Pegar os últimos 60 valores
        if len(close_prices) < WINDOW_SIZE:
            raise HTTPException(status_code=400,
                              detail=f"Dados insuficientes. Necessário {WINDOW_SIZE} dias, mas obteve apenas {len(close_prices)}.")
        
        last_60_prices = close_prices[-WINDOW_SIZE:].tolist()
        
        # 5. Fazer a previsão usando a mesma lógica do endpoint POST
        input_array = np.array(last_60_prices).reshape(-1, 1)
        scaled_input = scaler.transform(input_array)
        reshaped_input = np.reshape(scaled_input, (1, WINDOW_SIZE, 1))
        
        prediction_scaled = model.predict(reshaped_input)
        prediction = scaler.inverse_transform(prediction_scaled)
        
        # 6. Retornar dados históricos + previsão
        return {
            "codigo_acao": codigo_acao.upper(),
            "historical_prices": last_60_prices,
            "predicted_next_day_close_price": float(prediction[0][0]),
            "last_known_date": df.index[-1].strftime('%Y-%m-%d'),
            "prediction_date": (df.index[-1] + timedelta(days=1)).strftime('%Y-%m-%d')
        }
    
    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno durante a previsão: {str(e)}")