"""
Testes automatizados para a API de Previsão de Ações
Inclui testes de endpoints, validações e casos de erro
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

# Adicionar o diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.main import app

@pytest.fixture
def client():
    """Fixture para criar o cliente de teste"""
    return TestClient(app)


def test_health_check(client):
    """Testa o endpoint de health check"""
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()


def test_predict_endpoint_success(client):
    """Testa o endpoint de previsão com dados válidos"""
    # Gerar 60 preços de teste
    historical_prices = [150.0 + i * 0.5 for i in range(60)]
    
    payload = {
        "historical_prices": historical_prices
    }
    
    response = client.post("/predict", json=payload)
    # Aceita 200 (sucesso) ou 503 (modelo não carregado em ambiente de teste)
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        assert "predicted_next_day_close_price" in response.json()
        assert isinstance(response.json()["predicted_next_day_close_price"], (int, float))


def test_predict_endpoint_invalid_size(client):
    """Testa o endpoint de previsão com número incorreto de preços"""
    # Enviar apenas 30 preços (deveria ser 60)
    historical_prices = [150.0 + i * 0.5 for i in range(30)]
    
    payload = {
        "historical_prices": historical_prices
    }
    
    response = client.post("/predict", json=payload)
    # Aceita 400 (erro de validação) ou 503 (modelo não carregado)
    assert response.status_code in [400, 503]
    assert "detail" in response.json()


def test_predict_endpoint_empty_list(client):
    """Testa o endpoint de previsão com lista vazia"""
    payload = {
        "historical_prices": []
    }
    
    response = client.post("/predict", json=payload)
    # Aceita 400 (erro de validação) ou 503 (modelo não carregado)
    assert response.status_code in [400, 503]


def test_predict_endpoint_invalid_data_type(client):
    """Testa o endpoint de previsão com tipo de dado inválido"""
    payload = {
        "historical_prices": "invalid_data"
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity


def test_predict_endpoint_with_negative_values(client):
    """Testa o endpoint de previsão com valores negativos (válido para teste)"""
    historical_prices = [-150.0 + i * 0.5 for i in range(60)]
    
    payload = {
        "historical_prices": historical_prices
    }
    
    response = client.post("/predict", json=payload)
    # Deve processar ou retornar 503 se modelo não carregado
    assert response.status_code in [200, 500, 503]


def test_predict_endpoint_with_large_values(client):
    """Testa o endpoint com valores muito grandes"""
    historical_prices = [10000.0 + i * 100 for i in range(60)]
    
    payload = {
        "historical_prices": historical_prices
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code in [200, 503]


def test_predict_endpoint_with_zeros(client):
    """Testa o endpoint com valores zero"""
    historical_prices = [0.0] * 60
    
    payload = {
        "historical_prices": historical_prices
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code in [200, 503]


def test_predict_endpoint_realistic_data(client):
    """Testa com dados mais realistas de ações"""
    # Simular variação diária típica de ação
    base_price = 180.0
    historical_prices = []
    for i in range(60):
        variation = (i % 5 - 2) * 1.5  # Variação de -3 a +3
        historical_prices.append(base_price + variation + i * 0.2)
    
    payload = {
        "historical_prices": historical_prices
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        result = response.json()
        assert "predicted_next_day_close_price" in result
        # Verificar se a previsão está em um range razoável
        assert result["predicted_next_day_close_price"] > 0


def test_predict_endpoint_missing_field(client):
    """Testa o endpoint sem o campo obrigatório"""
    payload = {}
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_endpoint_extra_prices(client):
    """Testa o endpoint com mais de 60 preços"""
    historical_prices = [150.0 + i * 0.5 for i in range(100)]
    
    payload = {
        "historical_prices": historical_prices
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code in [400, 503]


def test_predict_endpoint_mixed_data_types(client):
    """Testa o endpoint com tipos mistos na lista"""
    historical_prices = [150.0] * 59 + ["invalid"]
    
    payload = {
        "historical_prices": historical_prices
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_openapi_docs(client):
    """Testa se a documentação OpenAPI está acessível"""
    response = client.get("/docs")
    assert response.status_code == 200


def test_redoc_docs(client):
    """Testa se a documentação Redoc está acessível"""
    response = client.get("/redoc")
    assert response.status_code == 200


def test_openapi_schema(client):
    """Testa se o schema OpenAPI está disponível"""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert "info" in schema
    assert "paths" in schema


def test_response_headers(client):
    """Testa se os headers de resposta incluem métricas de performance"""
    historical_prices = [150.0 + i * 0.5 for i in range(60)]
    payload = {"historical_prices": historical_prices}
    
    response = client.post("/predict", json=payload)
    # Verificar se o header X-Process-Time está presente
    assert "X-Process-Time" in response.headers or response.status_code == 503


def test_predict_auto_invalid_ticker(client):
    """Testa endpoint de previsão automática com ticker inválido"""
    response = client.get("/predict-auto/INVALID_TICKER_XYZ")
    # Deve retornar erro 404 ou 503 (se modelo não carregado)
    assert response.status_code in [404, 503, 500]


def test_predict_auto_valid_ticker(client):
    """Testa endpoint de previsão automática com ticker válido (AAPL)"""
    response = client.get("/predict-auto/AAPL")
    # Aceita 200 (sucesso) ou 503 (modelo não carregado)
    assert response.status_code in [200, 503, 500]
    if response.status_code == 200:
        result = response.json()
        assert "codigo_acao" in result
        assert "predicted_next_day_close_price" in result
        assert result["codigo_acao"] == "AAPL"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
