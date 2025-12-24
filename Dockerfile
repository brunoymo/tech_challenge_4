# 1. Imagem Base
# Usar uma imagem slim do Python para manter o tamanho reduzido
FROM python:3.11-slim

# 2. Definir o Diretório de Trabalho
WORKDIR /app

# 3. Variáveis de Ambiente
# Evita que o Python gere arquivos .pyc e armazene em buffer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=-1
ENV TF_CPP_MIN_LOG_LEVEL=2

# 4. Instalar Dependências
# Copiar apenas o requirements.txt primeiro para aproveitar o cache do Docker
COPY requirements.txt .

# Instalar as bibliotecas
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copiar o Código da Aplicação
# Copiar a pasta 'api' (que contém main.py e models/) para o contêiner
COPY ./api /app/api

# 6. Expor a Porta
EXPOSE 8000

# 7. Health Check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/')" || exit 1

# 8. Comando de Execução
# Iniciar o servidor Uvicorn quando o contêiner for executado
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]