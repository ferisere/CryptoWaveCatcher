# CryptoWaveCatcher

**CryptoWaveCatcher** is a Python tool designed to forecast "price waves" — rapid, significant price movements in cryptocurrencies. It combines market data from Bitfinex with technical indicators (ATR, VWAP) and social media trends (e.g., X activity) to predict waves using an RNN with an attention mechanism. The tool generates interactive visualizations with Holoviews, enabling traders to explore market opportunities with ease.

## Features
- Fetches real-time OHLCV data from Bitfinex.
- Incorporates social media trends for enhanced predictions.
- Uses an RNN with attention mechanism to detect price waves.
- Generates interactive visualizations with Holoviews.
- Configurable for various symbols and timeframes.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/CryptoWaveCatcher.git
   cd CryptoWaveCatcher
