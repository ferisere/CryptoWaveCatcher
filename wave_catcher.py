import ccxt
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, Attention
import holoviews as hv
from holoviews.operation.datashader import datashade
from datetime import datetime, timedelta
import os
hv.extension('bokeh')

class CryptoWaveCatcher:
    def __init__(self, symbol='XRP/USDT', timeframe='1h', lookback_days=15, api_key=None, api_secret=None):
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback_days = lookback_days
        self.exchange = ccxt.bitfinex({'apiKey': api_key, 'secret': api_secret})
        self.scaler = MinMaxScaler()

    def fetch_ohlcv(self):
        """Получение исторических данных с Bitfinex."""
        since = int((datetime.now() - timedelta(days=self.lookback_days)).timestamp() * 1000)
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, since=since)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def fetch_social_trends(self):
        """Получение социальных трендов с X (заглушка для API X)."""
        # Реальная версия требует API X для анализа трендов
        return np.random.rand(len(self.fetch_ohlcv())) * 80

    def calculate_atr(self, df, period=14):
        """Расчет ATR (Average True Range)."""
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    def calculate_vwap(self, df):
        """Расчет VWAP (Volume-Weighted Average Price)."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        return (typical_price * df['volume']).cumsum() / df['volume'].cumsum()

    def prepare_data(self, df):
        """Подготовка данных для модели."""
        df['returns'] = df['close'].pct_change()
        df['atr'] = self.calculate_atr(df)
        df['vwap'] = self.calculate_vwap(df)
        df['social_trends'] = self.fetch_social_trends()
        features = df[['close', 'atr', 'vwap', 'social_trends']].dropna()

        scaled_data = self.scaler.fit_transform(features)
        X, y = [], []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i])
            y.append(1 if scaled_data[i, 0] > np.percentile(scaled_data[:, 0], 90) else 0)  # Ценовая волна
        return np.array(X), np.array(y)

    def build_model(self):
        """Создание RNN-модели с механизмом внимания."""
        inputs = Sequential([
            SimpleRNN(64, return_sequences=True, input_shape=(60, 4)),
            Dropout(0.2),
            Attention(),
            SimpleRNN(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model = Sequential([inputs])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, X, y):
        """Обучение модели."""
        model = self.build_model()
        model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
        return model

    def predict_wave(self, model, X):
        """Прогноз ценовых волн."""
        predictions = model.predict(X)
        return (predictions > 0.5).astype(int)

    def visualize_results(self, df, predictions):
        """Визуализация с Holoviews."""
        df = df.iloc[60:].copy()
        df['wave_prediction'] = predictions

        curve = hv.Curve(df, 'timestamp', 'close', label='Price').opts(color='blue')
        points = hv.Scatter(df[df['wave_prediction'] == 1], 'timestamp', 'close', label='Predicted Wave').opts(color='red', size=10)
        plot = (curve * points).opts(title=f'Price Waves for {self.symbol}', width=800, height=400)

        hv.render(plot).save('data/sample_output/wave_visualization.html')

    def run(self):
        """Основной метод анализа."""
        df = self.fetch_ohlcv()
        X, y = self.prepare_data(df)
        model = self.train_model(X, y)
        predictions = self.predict_wave(model, X)
        self.visualize_results(df, predictions)
        print(f"Price waves predicted: {np.sum(predictions)} out of {len(predictions)} periods.")

if __name__ == "__main__":
    catcher = CryptoWaveCatcher(symbol='XRP/USDT', timeframe='1h', lookback_days=15)
    catcher.run()
