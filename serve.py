import os
from dotenv import load_dotenv

import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import logging
from time import perf_counter
import curl_cffi.requests.impersonate as curl_impersonate
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
import psutil

load_dotenv()  # carrega variáveis de ambiente locais (.env) para testes

sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    traces_sample_rate=1.0,
    profile_session_sample_rate=1.0,
    environment=os.getenv("SENTRY_ENVIRONMENT"),
    enable_logs=True,
    send_default_pii=True,
    profile_lifecycle="trace",
)

# Paths for artifacts
ARTEFACT_DIR = Path("artifacts")
METADATA_PATH = ARTEFACT_DIR / "metadata.json"
MODEL_PATH = ARTEFACT_DIR / "lstm_model.keras"
SCALER_PATH = ARTEFACT_DIR / "scaler.pkl"
CALIBRATOR_PATH = ARTEFACT_DIR / "calibrator.pkl"

# Ajusta fingerprint padrão do curl_cffi para evitar ImpersonateError (chrome136 não suportado aqui)
curl_impersonate.DEFAULT_CHROME = "chrome120"
curl_impersonate.REAL_TARGET_MAP["chrome"] = "chrome120"

logger = logging.getLogger("predict_live")
YF_CACHE_DIR = ARTEFACT_DIR / "yf_cache"
YF_CACHE_DIR.mkdir(exist_ok=True, parents=True)
try:
    # Evita erros de "readonly database" em cache padrão
    yf.set_tz_cache_location(str(YF_CACHE_DIR))
    from yfinance.cache import _TzDBManager
    _TzDBManager.set_location(str(YF_CACHE_DIR))
except Exception as exc:
    logger.warning("Falha ao configurar cache do yfinance: %s", exc)

class PredictRequest(BaseModel):
    """
    Corpo da requisição para previsão.
    """

    rows: List[Dict[str, float]] = Field(
        ...,
        description="Lista cronológica com pelo menos 20 registros, contendo as colunas esperadas (veja /health).",
        json_schema_extra={
            "example": {
                "rows": [
                    {
                        "open_amzn": 155.02,
                        "high_amzn": 156.71,
                        "low_amzn": 154.31,
                        "close_amzn": 155.90,
                        "volume_amzn": 72042000,
                        "return_1d": 0.0012,
                        "return_5d": 0.0098,
                        "ma_5": 155.4,
                        "ma_10": 154.9,
                        "ma_20": 154.2,
                        "vol_10": 0.011,
                        "price_range": 0.014,
                        "volume_trend": 0.02,
                    }
                ]
                * 20,
            }
        },
    )


def load_artifacts():
    if not METADATA_PATH.exists():
        raise RuntimeError("metadata.json não encontrado em artifacts/")
    metadata = json.loads(METADATA_PATH.read_text())
    feature_order = metadata["feature_order"]
    lookback = metadata["lookback"]
    close_idx = metadata["close_idx"]

    scaler = joblib.load(SCALER_PATH)
    calibrator = joblib.load(CALIBRATOR_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)
    return model, scaler, calibrator, feature_order, lookback, close_idx


model, scaler, calibrator, FEATURE_ORDER, LOOKBACK, CLOSE_IDX = load_artifacts()
app = FastAPI(
    title="LSTM Forecast API",
    version="0.1.0",
    description=(
        "API de demonstração para previsão do próximo preço (AMZN) com LSTM.\\n"
        "- Envie pelo menos 20 linhas em ordem cronológica.\\n"
        "- As colunas esperadas estão em /health."
    ),
)

PROCESS = psutil.Process()


@app.middleware("http")
async def sentry_resource_metrics(request, call_next):
    start = perf_counter()
    cpu_start = PROCESS.cpu_times()
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        # CPU em milissegundos e memória em MB no fim da requisição
        cpu_end = PROCESS.cpu_times()
        cpu_ms = (cpu_end.user - cpu_start.user + cpu_end.system - cpu_start.system) * 1000
        mem_mb = PROCESS.memory_info().rss / (1024 * 1024)
        duration_ms = (perf_counter() - start) * 1000
        path = request.url.path
        method = request.method
        status = getattr(response, "status_code", 500)

        sentry_sdk.set_measurement("request.duration", duration_ms, unit="millisecond")
        sentry_sdk.set_measurement("request.cpu", cpu_ms, unit="millisecond")
        sentry_sdk.set_measurement("request.mem", mem_mb, unit="megabyte")
        with sentry_sdk.configure_scope() as scope:
            scope.set_tag("http.path", path)
            scope.set_tag("http.method", method)
            scope.set_tag("http.status_code", status)


def invert_scale(values: np.ndarray) -> np.ndarray:
    zeros = np.zeros((len(values), len(FEATURE_ORDER)))
    zeros[:, CLOSE_IDX] = values
    return scaler.inverse_transform(zeros)[:, CLOSE_IDX]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduz as features usadas no treino."""
    # Normaliza nomes vindos do yfinance (multiindex ou não) e aceita upper/lower
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in tup if x is not None and str(x) != ""])
            .lower()
            .replace(" ", "_")
            for tup in df.columns
        ]
    else:
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

    rename_upper = {
        "open": "open_amzn",
        "high": "high_amzn",
        "low": "low_amzn",
        "close": "close_amzn",
        "volume": "volume_amzn",
    }
    # Já está no formato esperado?
    base_cols = ["open_amzn", "high_amzn", "low_amzn", "close_amzn", "volume_amzn"]
    if set(base_cols).issubset(df.columns):
        pass
    elif set(rename_upper.keys()).issubset(df.columns):
        df = df.rename(columns=rename_upper)
    else:
        raise ValueError(f"Colunas base ausentes: {base_cols}")
    df = df[base_cols].copy()

    df["return_1d"] = df["close_amzn"].pct_change()
    df["return_5d"] = df["close_amzn"].pct_change(5)
    df["ma_5"] = df["close_amzn"].rolling(5).mean()
    df["ma_10"] = df["close_amzn"].rolling(10).mean()
    df["ma_20"] = df["close_amzn"].rolling(20).mean()
    df["vol_10"] = df["close_amzn"].pct_change().rolling(10).std()
    df["price_range"] = (df["high_amzn"] - df["low_amzn"]) / df["close_amzn"]
    df["volume_trend"] = df["volume_amzn"].pct_change()

    df = df.dropna()
    # Garante a ordem correta das colunas
    df = df[FEATURE_ORDER]
    return df


def predict_from_dataframe(df_feat: pd.DataFrame):
    """Aplica scaler, monta janela e gera previsões (normalizada, preço, calibrada)."""
    if len(df_feat) < LOOKBACK:
        raise ValueError(f"Dados insuficientes: {len(df_feat)} linhas de features, requer {LOOKBACK}.")

    df_feat = df_feat.tail(LOOKBACK)
    scaled = scaler.transform(df_feat.values)
    X = scaled.reshape(1, LOOKBACK, len(FEATURE_ORDER))

    y_pred_norm = model.predict(X).squeeze()
    y_pred_price = invert_scale(np.array([y_pred_norm]))[0]
    y_pred_price_cal = calibrator.predict(np.array([[y_pred_price]]))[0]

    return y_pred_norm, y_pred_price, y_pred_price_cal


@app.get(
    "/health",
    summary="Status e esquema de entrada",
    response_description="Estado do serviço e metadados de entrada",
)
def health():
    return {"status": "ok", "lookback": LOOKBACK, "features": FEATURE_ORDER}


@app.post(
    "/predict",
    summary="Prever próximo preço",
    response_description="Previsão normalizada, em preço e calibrada",
)
def predict(req: PredictRequest):
    if len(req.rows) < LOOKBACK:
        raise HTTPException(
            status_code=400, detail=f"É necessário pelo menos {LOOKBACK} registros."
        )

    df = pd.DataFrame(req.rows)
    missing = set(FEATURE_ORDER) - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Faltam colunas: {sorted(missing)}. Esperado: {FEATURE_ORDER}",
        )

    df = df[FEATURE_ORDER].tail(LOOKBACK)
    y_pred_norm, y_pred_price, y_pred_price_cal = predict_from_dataframe(df)

    return {
        "prediction_normalized": float(y_pred_norm),
        "prediction_price": float(y_pred_price),
        "prediction_price_calibrated": float(y_pred_price_cal),
        "lookback_used": LOOKBACK,
    }

@app.get(
    "/predict_live",
    summary="Prever usando dados recentes do yfinance",
    response_description="Previsão normalizada, em preço e calibrada",
)
def predict_live(ticker: str = "AMZN"):
    """
    Busca dados recentes do yfinance, gera features e retorna previsão usando somente dados online.
    - `ticker`: símbolo (default AMZN).
    """
    # Tenta baixar do yfinance com a mesma configuração do treino_notbook
    end_date = datetime.now(timezone.utc).date()
    # Baixa mais dias corridos para compensar dias sem pregão e perdas por rolling/dropna
    start_date = end_date - timedelta(days=max(LOOKBACK * 4, 90))

    df_raw = yf.download(
        tickers=ticker,
        start=start_date.isoformat(),
        end=(end_date + timedelta(days=1)).isoformat(),  # end é exclusivo
        interval="1d",
        auto_adjust=True,
        actions=True,
        group_by="column",
        progress=False,
        threads=True,
    )
   
    if df_raw is None or df_raw.empty:
        logger.error(
            "yfinance retornou vazio",
            extra={"ticker": ticker, "start": start_date.isoformat(), "end": end_date.isoformat()},
        )
        raise HTTPException(status_code=500, detail="Sem dados retornados do yfinance.")

    try:
        # Normaliza colunas como no notebook antes de gerar features
        df_feat = build_features(df_raw)
        y_pred_norm, y_pred_price, y_pred_price_cal = predict_from_dataframe(df_feat)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "ticker": ticker,
        "rows_used": LOOKBACK,
        "prediction_normalized": float(y_pred_norm),
        "prediction_price": float(y_pred_price),
        "prediction_price_calibrated": float(y_pred_price_cal),
        "features_rows_available": int(len(df_feat)),
        "source": "yfinance",
    }

if __name__ == "__main__":

    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=False)
