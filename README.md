# LSTM Forecast API
API FastAPI que serve um modelo LSTM treinado para prever o próximo preço (AMZN por padrão). Usa artefatos prontos salvos em `artifacts/` (modelo, scaler, calibrador e metadados).

## Requisitos
- Python 3.10+ recomendado
- Dependências: `pip install -r requirements.txt`
- Artefatos necessários em `artifacts/`: `lstm_model.keras`, `scaler.pkl`, `calibrator.pkl`, `metadata.json`

## Como rodar local
```bash
uvicorn serve:app --host 0.0.0.0 --port 8000
```
Endpoints:
- `/health` – status e lista de features esperadas.
- `/predict` – POST com JSON contendo `rows` (mínimo LOOKBACK linhas com as colunas de `FEATURE_ORDER`).
- `/predict_live` – GET, baixa dados do yfinance para o ticker (default AMZN), gera features e retorna previsão.

## Estrutura principal
- `serve.py`: carrega artefatos, monta features, aplica scaler, prediz e calibra. Expõe rotas FastAPI.
- `artifacts/`: artefatos de treino necessários para servir.
- `treino_notbook.ipynb`: notebook de treino (fonte dos artefatos).

## Deploy (ex.: Railway)
- Inclua `requirements.txt` e `Procfile` com `web: uvicorn serve:app --host 0.0.0.0 --port $PORT`.
- Commit/push para o GitHub e conecte o repositório no Railway; configure a branch de deploy.
- Certifique-se de que `artifacts/` está versionada ou disponível em um Volume.
