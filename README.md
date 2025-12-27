# Projeto de Previsão de Ações com LSTM e FastAPI

Este projeto implementa um pipeline completo de MLOps para prever o preço de fechamento de ações usando uma Rede Neural LSTM. O modelo é treinado e implantado como uma API RESTful usando FastAPI e Docker.

## Links do Projeto

- **API em Produção**: https://tech-challenge-4.onrender.com
- **Documentação da API**: https://tech-challenge-4.onrender.com/docs
- **Repositório Git**: https://github.com/seu-usuario/tech-challenge

## Estrutura do Projeto

```
tech_challenge/
├── api/
│   ├── __init__.py
│   ├── main.py              # Lógica da API FastAPI
│   └── models/
│       ├── stock_lstm_model.h5  # Modelo salvo (gerado após treino)
│       └── scaler.pkl           # Escalonador salvo (gerado após treino)
├── .dockerignore
├── Dockerfile               # Definição do contêiner da API
├── docker-compose.yml       # Orquestração de containers
├── README.md                # Esta documentação
├── requirements.txt         # Dependências Python
└── train_model.py          # Script de coleta e treinamento
```

## Instalação

### 1. Clone o repositório

```bash
git clone <url-do-seu-repositorio>
cd tech_challenge
```

### 2. Crie um ambiente virtual e instale as dependências

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Treinamento do Modelo

Antes de executar a API, você deve treinar o modelo.

```bash
python train_model.py
```

Este script irá:
- Baixar dados históricos de ações (ou criar dados sintéticos se houver falha)
- Treinar o modelo LSTM
- Salvar `stock_lstm_model.h5` e `scaler.pkl` em `api/models/`

## Executando a API

### A. Localmente (desenvolvimento)

Execute o servidor Uvicorn:

```bash
uvicorn api.main:app --reload
```

A API estará disponível em:
- **API:** http://127.0.0.1:8000
- **Documentação Interativa (Swagger):** http://127.0.0.1:8000/docs
- **Redoc:** http://127.0.0.1:8000/redoc

### B. Via Docker

#### Opção 1: Docker Compose (Recomendado)

```bash
docker-compose up -d
```

Para parar:
```bash
docker-compose down
```

#### Opção 2: Docker Build Manual

1. Construa a imagem:
```bash
docker build -t stock-prediction-api .
```

2. Execute o container:
```bash
docker run -d -p 8000:8000 --name stock-api stock-prediction-api
```

3. Verifique os logs:
```bash
docker logs stock-api
```

4. Para parar:
```bash
docker stop stock-api
docker rm stock-api
```

## Como Usar a API

### Health Check

```bash
curl http://127.0.0.1:8000/
```

**Resposta:**
```json
{
  "status": "API está funcionando e o modelo está carregado."
}
```

### Fazer Previsão

**Requisição POST** para `/predict` com os últimos 60 preços de fechamento:

**PowerShell:**
```powershell
$body = @{
    historical_prices = @(
        190.10, 191.20, 190.50, 192.00, 193.10, 192.80, 194.50, 195.00, 196.20, 195.90,
        197.10, 198.20, 197.50, 199.00, 200.10, 199.80, 201.50, 202.00, 203.20, 202.90,
        204.10, 205.20, 204.50, 206.00, 207.10, 206.80, 208.50, 209.00, 210.20, 209.90,
        211.10, 212.20, 211.50, 213.00, 214.10, 213.80, 215.50, 216.00, 217.20, 216.90,
        218.10, 219.20, 218.50, 220.00, 221.10, 220.80, 222.50, 223.00, 224.20, 223.90,
        225.10, 226.20, 225.50, 227.00, 228.10, 227.80, 229.50, 230.00, 231.20, 230.90
    )
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -Body $body -ContentType "application/json"
```

**curl:**
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "historical_prices": [
    190.10, 191.20, 190.50, 192.00, 193.10, 192.80, 194.50, 195.00, 196.20, 195.90,
    197.10, 198.20, 197.50, 199.00, 200.10, 199.80, 201.50, 202.00, 203.20, 202.90,
    204.10, 205.20, 204.50, 206.00, 207.10, 206.80, 208.50, 209.00, 210.20, 209.90,
    211.10, 212.20, 211.50, 213.00, 214.10, 213.80, 215.50, 216.00, 217.20, 216.90,
    218.10, 219.20, 218.50, 220.00, 221.10, 220.80, 222.50, 223.00, 224.20, 223.90,
    225.10, 226.20, 225.50, 227.00, 228.10, 227.80, 229.50, 230.00, 231.20, 230.90
  ]
}'
```

**Resposta:**
```json
{
  "predicted_next_day_close_price": 232.15
}
```

## Testando

Acesse a documentação interativa em:
- http://127.0.0.1:8000/docs

E teste diretamente pela interface Swagger.

## Tecnologias Utilizadas

- **Python 3.12**
- **TensorFlow/Keras** - Modelo LSTM
- **FastAPI** - Framework Web
- **Uvicorn** - Servidor ASGI
- **scikit-learn** - Pré-processamento
- **yfinance** - Coleta de dados
- **Docker** - Containerização

## Configurações

### Variáveis de Ambiente

- `PYTHONUNBUFFERED=1` - Desabilita buffering do Python
- `TF_ENABLE_ONEDNN_OPTS=0` - Desabilita otimizações oneDNN (opcional)

### Parâmetros do Modelo

No `train_model.py`:
- `WINDOW_SIZE = 60` - Janela de 60 dias
- `TICKER = 'AAPL'` - Ação a ser prevista
- `EPOCHS = 50` - Épocas de treinamento

## Troubleshooting

### Erro: "Modelo ou escalonador não estão carregados"
- Execute `python train_model.py` primeiro

### Erro: "A entrada deve conter exatamente 60 preços históricos"
- Certifique-se de enviar exatamente 60 valores no array `historical_prices`

### Porta 8000 já em uso
```bash
# Encontrar processo usando a porta
netstat -ano | findstr :8000

# Parar o processo (Windows)
taskkill /PID <PID> /F
```

## Resultados e Métricas do Modelo

O modelo LSTM foi treinado com dados históricos da ação AAPL (Apple Inc.) de 2018 a 2024.

### Arquitetura do Modelo
- **2 camadas LSTM** com 64 unidades cada
- **Dropout layers** (0.2) para regularização
- **Camadas Dense** para saída
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)
- **Window Size**: 60 dias

### Métricas de Avaliação (Conjunto de Teste)

```
Mean Absolute Error (MAE):           4.75
Root Mean Squared Error (RMSE):      5.75
Mean Absolute Percentage Error (MAPE): 2.55%
```

### Interpretação dos Resultados

- **MAE de 4.75**: O modelo erra em média $4.75 na previsão do preço de fechamento
- **RMSE de 5.75**: Penaliza erros maiores, indicando que a maioria das previsões está próxima do valor real
- **MAPE de 2.55%**: Erro percentual médio muito bom, indicando alta precisão relativa

### Performance em Produção

A API está deployada no Render.com com as seguintes características:
- **Tempo de resposta médio**: < 500ms
- **Cold start**: 30-60 segundos (plano free)
- **Disponibilidade**: 99%+ (com sleep após inatividade no plano free)

## API em Produção

**URL Base**: https://tech-challenge-4.onrender.com

**Endpoints Disponíveis**:
- Health Check: `GET /`
- Documentação: `GET /docs`
- Previsão Manual: `POST /predict`
- Previsão Automática: `GET /predict-auto/{codigo_acao}`

**Exemplo de Uso**:
```bash
curl https://tech-challenge-4.onrender.com/predict-auto/AAPL
```

## Licença

Este projeto foi desenvolvido como parte do Tech Challenge - FIAP Pós-Tech Machine Learning Engineering.

## Autores

Grupo 35 - FIAP Pós-Tech Machine Learning Engineering - Turma 2024
