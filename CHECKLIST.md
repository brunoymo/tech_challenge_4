# Guia Completo do Projeto - Tech Challenge Fase 4

## 📋 Checklist de Implementação

### ✅ 1. Coleta e Pré-processamento dos Dados
- [x] Biblioteca yfinance implementada
- [x] Download de dados históricos (AAPL 2018-2024)
- [x] Normalização com MinMaxScaler (0-1)
- [x] Criação de janelas deslizantes (60 dias)
- [x] Divisão treino/teste (80/20)

### ✅ 2. Desenvolvimento do Modelo LSTM
- [x] Arquitetura LSTM implementada:
  - 2 camadas LSTM (64 unidades cada)
  - Camadas Dropout (0.2) para evitar overfitting
  - Camadas Dense para saída
- [x] Treinamento configurado:
  - 50 epochs
  - Batch size: 32
  - Optimizer: Adam
  - Loss: MSE
- [x] Métricas de avaliação:
  - MAE: 4.75
  - RMSE: 5.75
  - MAPE: 2.55%

### ✅ 3. Salvamento e Exportação do Modelo
- [x] Modelo salvo em formato HDF5 (.h5)
- [x] Scaler salvo em formato pickle (.pkl)
- [x] Artefatos em `api/models/`

### ✅ 4. Deploy do Modelo
- [x] API RESTful com FastAPI
- [x] Endpoints implementados:
  - `GET /` - Health check
  - `POST /predict` - Previsão com dados manuais
  - `GET /predict-auto/{codigo_acao}` - Previsão automática
- [x] Documentação Swagger automática
- [x] Containerização com Docker

### ✅ 5. Escalabilidade e Monitoramento
- [x] Middleware de tempo de resposta
- [x] Health checks no Docker
- [x] Logs estruturados
- [x] Testes automatizados (pytest)

## 📊 Resultados do Modelo

```
Mean Absolute Error (MAE):    4.75
Root Mean Squared Error (RMSE): 5.75
Mean Absolute Percentage Error (MAPE): 2.55%
```

## 🚀 Como Executar

### Localmente
```bash
python train_model.py
uvicorn api.main:app --reload
```

### Docker
```bash
docker-compose up -d
```

## 📝 Estrutura do Projeto

```
tech_challenge/
├── api/
│   ├── __init__.py
│   ├── main.py              # API FastAPI
│   └── models/
│       ├── stock_lstm_model.h5
│       └── scaler.pkl
├── .dockerignore
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── README.md
├── requirements.txt
├── test_api.py
└── train_model.py
```

## 🔗 Links Importantes

- **Documentação da API (Local)**: http://127.0.0.1:8000/docs
- **Repositório Git**: [URL do seu repositório]
- **Deploy em Produção**: [URL se houver]

## 👥 Autores

Grupo 35 - FIAP Pós-Tech MLET

## 📄 Licença

Este projeto foi desenvolvido como parte do Tech Challenge - FIAP Pós-Tech Machine Learning Engineering.
