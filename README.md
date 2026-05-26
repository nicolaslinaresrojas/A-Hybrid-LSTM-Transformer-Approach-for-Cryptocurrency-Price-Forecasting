> **A Parallel Hybrid LSTM-Transformer Approach with Dynamic Directional Volatility-Adjusted Loss for Multi-Asset, Multi-Temporal Cryptocurrency Forecasting**

*Universidad Industrial de Santander (UIS) — Daniel Aguilar Navas · Ivan Augusto Camargo López · Nicolas Linares Rojas*

---

## 📌 Overview

This repository contains the full implementation of a novel deep learning framework for cryptocurrency price forecasting. The architecture combines **Bidirectional LSTMs** and **Transformer Encoders** in a parallel dual-stream design, fused through a **Bidirectional Cross-Attention** interaction layer.

Key innovations include:

- **Parallel Hybrid LSTM/GRU-Transformer** architecture that simultaneously captures local sequential patterns and global long-range dependencies.
- **Directional Volatility-Adjusted (DVA) Loss** — a custom loss function coupling Huber-based magnitude estimation with a continuous directional penalty, dynamically scaled by the real-time **VIX index**.
- **PACF-based context autotuning** using the Box-Jenkins methodology to determine the optimal look-back window from market volatility statistics.
- **Chronological Walk-Forward Validation** across Daily, Hourly, and 15-Minute temporal resolutions to prevent data leakage.

The framework was evaluated on **Bitcoin (BTC)**, **Ethereum (ETH)**, and **Solana (SOL)**, achieving consistent Mean Directional Accuracy (MDA) above the random-walk baseline (>50%).

---

## 🏗️ Architecture

<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/375c7170-2dd7-4d3e-851c-1952a8b6c4a7" />


The **Transformer stream** captures macro/global dependencies without distance-based degradation. The **BiLSTM stream** reinforces local temporal inductive biases. Cross-attention allows both streams to mutually enrich each other's representations before fusion.

---

## 📊 Results Summary

### Standard Loss vs. DVA Loss (Bitcoin)

| Timeframe | Model | RMSE (Std) | RMSE (DVA) | MDA (DVA) |
|-----------|-------|-----------|-----------|----------|
| Daily | Hybrid LSTM | $3,842.28 | **$2,928.20** | 51.53% |
| Daily | Hybrid GRU | $3,369.85 | $2,928.38 | 51.45% |
| Hourly | Hybrid GRU | $1,625.23 | $1,534.87 | 51.04% |
| 15-Min | Hybrid GRU | $399.71 | **$351.03** | 51.15% |

The DVA loss reduces RMSE by up to **~24–26%** compared to standard training objectives across all assets and timeframes.

### Optimized Framework (PACF + Dynamic VIX Loss, H=1)

| Timeframe | Best Model | RMSE | MAE | MDA |
|-----------|-----------|------|-----|-----|
| Daily | Hybrid GRU | $1,646.21 | $1,042.63 | 50.29% |
| Hourly | Hybrid GRU | $723.68 | $503.45 | 50.44% |
| 15-Min | Hybrid LSTM | $179.76 | $121.26 | **51.32%** |

---

## 📁 Repository Structure

```
├── Proyecto_IA_III_Final (1).ipynb   # Main implementation (Colab notebook export)
├── README.md
└── A Hybrid LSTM-Transformer Approach for Cryptocurrency Price Forecasting.pdf
```

The main script covers the full pipeline end-to-end:

1. Data extraction and feature engineering
2. Dimensionality reduction via XGBoost + SHAP
3. Walk-forward data splitting
4. Model definitions (Hybrid LSTM-Transformer, Hybrid GRU-Transformer, baselines)
5. DVA loss function
6. Training loop with walk-forward validation
7. Evaluation and visualization

---

## 🧰 Dependencies

```bash
pip install torch pandas numpy yfinance pandas-ta scikit-learn xgboost shap matplotlib seaborn statsmodels
```

| Library | Purpose |
|---------|---------|
| `torch` | Model definition and training |
| `yfinance` | Market data extraction |
| `pandas-ta` | Technical indicator computation |
| `xgboost` + `shap` | Feature selection via SHAP importance |
| `scikit-learn` | Preprocessing scalers and metrics |
| `statsmodels` | PACF computation for context autotuning |

---

## 🚀 Usage

The project is designed to run in **Google Colab** (GPU recommended). Open the notebook link or run the exported script:

```bash
python proyecto_ia_iii_final.py
```

The pipeline will automatically:
- Download multi-asset, multi-timeframe data from Yahoo Finance
- Compute technical indicators and scale features
- Select features via XGBoost + SHAP
- Train and evaluate all model variants under walk-forward validation
- Output RMSE, MAE, MAPE, and MDA metrics per timeframe

> ⚠️ Yahoo Finance free-tier limits: ~3118 daily, ~15316 hourly, and ~5413 15-minute candles for BTC-USD.

---

## 📐 Methodology

### Data & Features

- **Target:** BTC-USD Close price (log-returns)
- **Macro:** S&P 500, NASDAQ, VIX, US Dollar Index, US 10Y Treasury Yield
- **Commodities:** Gold, Crude Oil
- **Crypto context:** Ethereum (ETH-USD)
- **Technical indicators:** SMA-200, EMA-9/200, VWAP, RSI-14, MACD, MFI-60, WaveTrend (WT1/WT2)

### Scaling Protocol

| Feature Group | Scaler |
|--------------|--------|
| BTC Close, Volume, Equities, Forex, ETH, Trend indicators | StandardScaler |
| VIX, Crude Oil, MACD | RobustScaler |
| RSI, MFI, WaveTrend | MinMaxScaler \[-1, 1\] |

### DVA Loss Function

$$\mathcal{L}_{DVA}(y, \hat{y}) = \mathcal{L}_{Huber}(y, \hat{y};\delta) + \lambda \mathcal{L}_{dir}(y, \hat{y})$$

where the directional penalty term is:

$$\mathcal{L}_{dir}(y, \hat{y}) = \frac{1}{H}\sum_{h=1}^{H} \max\left(0, -(\hat{y}_h \cdot \text{sgn}(y_h))\right)$$

With dynamic VIX scaling:

$$\lambda_t = \lambda_{base} + VIX_{gt} \cdot (\lambda_{max} - \lambda_{base})$$

### Walk-Forward Validation

- **Initial warm-up:** 70% of data (primary train set)
- **Fine-tuning:** 15% (2nd train / recalibration)
- **Test:** 15% (blind, out-of-sample)
- Scalers are **re-fit** at each forward step to prevent look-ahead bias.

---

## 📈 Live Backtesting (TradingView)

The directional signals were exported as Pine Script strategies for realistic backtesting with 0.1% maker/taker commissions:

- **Bitcoin Daily:** https://es.tradingview.com/script/maAO3eJi/
- **Ethereum:** https://es.tradingview.com/script/gOE57uUb/

> Note: While MDA > 50% confirms a statistically quantified edge, backtesting with transaction costs yielded negative net returns — highlighting that directional accuracy alone is not sufficient for profitability without an advanced execution strategy.

---

## ⚙️ Hyperparameters

| Category | Parameter | Value |
|---------|----------|-------|
| Architecture | Model Dimension ($d_{model}$) | 32 |
| | Attention Heads | 4 |
| | Recurrent Hidden Size | 16 |
| | Dropout Rate | 0.2 |
| Optimization | Optimizer | Adam |
| | Base Learning Rate | 0.001 |
| | Weight Decay (L2) | 1e-4 |
| | Batch Size | 64 |
| | Epochs | 50 |
| DVA Loss | Huber Delta ($\delta$) | 1.0 |
| | Base Penalty ($\lambda_{base}$) | 0.1 |
| | Max Penalty ($\lambda_{max}$) | 1.5 |

---

## 🔬 Ablation Studies

| Experiment | Key Finding |
|-----------|------------|
| Heuristic sequence length (14→60) | Longer context reduces RMSE; peak MDA at S=30 |
| Dynamic VIX-Adjusted Loss | Higher MDA at cost of slightly higher RMSE — favorable trade-off |
| PACF Autotuning (Box-Jenkins) | Data-driven window selection further improves MDA and eliminates heuristic bias |
| Sinusoidal vs. Learnable PE | Sinusoidal encoding proved more robust across timeframes |
| Cross-asset generalization | Significant MDA degradation on ETH and SOL due to higher idiosyncratic volatility |

---

## 📄 Paper

This work was submitted as a research paper following the **CVPR 2026** format. If you use this code or methodology, please cite:

```bibtex
@article{aguilar2025hybridlstm,
  title     = {A Hybrid LSTM-Transformer Approach for Cryptocurrency Price Forecasting},
  author    = {Aguilar Navas, Daniel and Camargo L\'opez, Ivan Augusto and Linares Rojas, Nicolas},
  institution = {Universidad Industrial de Santander (UIS)},
  year      = {2025}
}
```

---

## 🛣️ Future Work

- **Long-Range Attention:** Explore Informer or Sparse Attention variants for processing high-frequency intra-minute ticks without quadratic memory overhead.
- **Sentiment Integration:** Incorporate NLP-based sentiment extraction from social media and macroeconomic event parsing.
- **Microstructural Data:** Integrate Limit Order Book (LOB) snapshots and liquidation cluster data for an institutional-grade informational edge.

---

## 📜 License

This project is for academic and research purposes. See individual library licenses for dependency terms.
