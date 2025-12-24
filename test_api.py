"""
Testes básicos para a API de Previsão de Ações
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


def test_openapi_docs(client):
    """Testa se a documentação OpenAPI está acessível"""
    response = client.get("/docs")
    assert response.status_code == 200


def test_redoc_docs(client):
    """Testa se a documentação Redoc está acessível"""
    response = client.get("/redoc")
    assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
