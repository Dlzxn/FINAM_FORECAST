import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_candles(csv_path, max_horizon=20, drop_cols=None, output_path=None):
    """
    Загружает CSV, создает day_1 … day_max_horizon как таргеты доходности,
    строит фичи, масштабирует их и возвращает готовый DataFrame.

    Формула:
        day_n = (close_{t+n} / close_t) - 1

    Args:
        csv_path (str): путь к исходному CSV.
        max_horizon (int): горизонт прогноза (например, 20).
        drop_cols (list): колонки, которые нужно удалить после фичеинженеринга.
        output_path (str, optional): путь для сохранения результата. Если None — не сохраняет.

    Returns:
        pd.DataFrame: обработанный и масштабированный DataFrame с таргетами.
    """
    df = pd.read_csv(csv_path)

    # --- Генерация таргетов day_1 … day_N ---
    for n in range(1, max_horizon + 1):
        df[f"day_{n}"] = (df["close"].shift(-n) / df["close"]) - 1

    # --- Базовые фичи ---
    df["price_change"] = abs(df["open"] - df["close"])
    df["absolute_change"] = abs(df["high"] - df["low"])
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # --- Волатильность ---
    df["vol_5d"] = df["log_return"].rolling(5).std()
    df["vol_20d"] = df["log_return"].rolling(20).std()
    df["vol_ratio"] = df["vol_5d"] / df["vol_20d"]

    # --- Свечной анализ ---
    df["oc_range"] = (df["close"] - df["open"]) / df["open"]
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]

    # --- Объём ---
    df["volume_rel"] = df["volume"] / df["volume"].rolling(20).mean()

    # --- Скользящие средние ---
    df["sma_5"] = df["close"].rolling(5).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_ratio"] = df["sma_5"] / df["sma_20"]

    # --- Временные признаки ---
    df["begin"] = pd.to_datetime(df["begin"])
    df["date_ts"] = df["begin"].astype("int64") // 10**9
    df["dayofweek"] = df["begin"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    # --- Категориальные признаки (тикеры) ---
    df = df.join(pd.get_dummies(df["ticker"], dtype=int))

    # --- Удаляем ненужные колонки ---
    default_drop = ["open", "close", "high", "low", "volume", "ticker", "absolute_change", "begin", "dayofweek"]
    if drop_cols is not None:
        default_drop += drop_cols
    df = df.drop(columns=[c for c in default_drop if c in df.columns])

    # --- Убираем пропуски (после шифта и rolling) ---
    df = df.dropna().reset_index(drop=True)

    # --- Разделяем таргеты ---
    target_cols = [f"day_{i}" for i in range(1, max_horizon + 1)]
    targets = df[target_cols]
    features = df.drop(columns=target_cols)

    # --- Масштабирование признаков ---
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    scaled_df = pd.DataFrame(scaled_features, columns=features.columns, index=features.index)

    # --- Собираем финальный датафрейм ---
    final_df = pd.concat([scaled_df, targets.reset_index(drop=True)], axis=1)

    if output_path:
        final_df.to_csv(output_path, index=False)

    return final_df


# Пример использования
if __name__ == "__main__":
    final_df = preprocess_candles(
        "../data/raw/participants/candles.csv",
        max_horizon=20,
        output_path="../data/raw/participants/train_candles.csv"
    )
    print(final_df.shape)