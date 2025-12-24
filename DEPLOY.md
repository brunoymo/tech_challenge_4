# Deploy no Render

## 🚀 Guia de Deploy

### Opção 1: Deploy Automático via GitHub (Recomendado)

1. **Acesse:** https://render.com
2. **Crie uma conta** (pode usar GitHub)
3. **Clique em "New +"** → **"Web Service"**
4. **Conecte seu repositório GitHub**
5. **Configure:**
   - **Name:** `stock-prediction-api`
   - **Environment:** `Docker`
   - **Plan:** `Free`
   - **Advanced:**
     - Health Check Path: `/`

6. **Clique em "Create Web Service"**

O Render vai:
- Detectar o `render.yaml` automaticamente
- Fazer o build do Docker
- Deployar a API

### Opção 2: Deploy Manual via Render.yaml

1. No Render Dashboard, clique em **"New +"** → **"Blueprint"**
2. Conecte o repositório
3. O Render vai detectar o `render.yaml` e configurar automaticamente

### ⏱️ Tempo de Deploy

- **Primeira vez:** 10-15 minutos (build do TensorFlow é pesado)
- **Updates posteriores:** 5-10 minutos

### 🔗 URLs Após Deploy

Após o deploy, você receberá uma URL tipo:
```
https://stock-prediction-api.onrender.com
```

**Endpoints:**
- Health: `https://stock-prediction-api.onrender.com/`
- Docs: `https://stock-prediction-api.onrender.com/docs`
- Previsão Auto: `https://stock-prediction-api.onrender.com/predict-auto/AAPL`

### ⚠️ Limitações do Plano Free

- **Sleep após inatividade:** API "dorme" após 15 minutos sem uso
- **Primeiro request:** Pode levar 30-60s (cold start)
- **RAM:** 512MB (suficiente para o modelo)
- **Build time:** 10-15 minutos

### 🔧 Troubleshooting

**Se o deploy falhar:**

1. **Verifique os logs** no Render Dashboard
2. **Problema comum:** Timeout no build
   - Solução: Render Free tem 10min de build. TensorFlow pode ultrapassar
   - Alternativa: Use plano pago temporariamente ou otimize o Docker

**Otimizar Docker (se necessário):**
```dockerfile
# Use imagem com TensorFlow pré-instalado
FROM tensorflow/tensorflow:latest-py3
```

### 📊 Monitoramento

No Render Dashboard você pode ver:
- Logs em tempo real
- Métricas de CPU/RAM
- Requests/segundo
- Status de saúde

### 💰 Upgrade para Produção

Para produção real, considere:
- **Plano Starter ($7/mês):** Sem sleep, mais RAM
- **Plano Pro:** Load balancing, mais recursos
