# Stefan Jansen ML4T (3rd Edition) Reference Mapping & XTrading Phase 14 Integration

*Reference Repository*: [stefan-jansen/machine-learning-for-trading](https://github.com/stefan-jansen/machine-learning-for-trading)  
*Purpose*: Extract quantitative research patterns, validation mechanisms, and cost/risk structures without copying raw strategy code into XTrading.

> [!IMPORTANT]
> **Phase 13 Protection**: Phase 13 prospective validation ($N=5/30$, PF `0.82`, -$45.41$) remains **100% frozen** and untouched. This mapping serves Phase 14 research only.

---

## 1. Module Reference Mapping Matrix

| ML4T Repository Module | Purpose & Core Mechanism | XTrading Equivalent | Reusable Mechanisms / Patterns | Modification Required for XTrading | Potential Risk | Phase 14 Relevance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `06_strategy_definition` | Pre-registering hypothesis, target, universe, and evaluation metrics before testing. | `phase14/configs/` | Structured research hypothesis specification & evidence boundaries. | Adapt for crypto perps (BTC & ETH 1h/4h). | Overfitting if hypothesis is changed after backtesting. | **HIGH** (Pre-registration) |
| `07_defining_the_learning_task` | Formulating prediction targets: continuous returns, directional classification, triple-barrier labels. | `phase14/labels/` | Signed 1H expected return vs. 3-class directional target. | Implement signed target $y_{t+1} = \frac{p_{t+1}-p_t}{p_t}$ and triple-barrier labeling. | Noise in 1H return labels. | **CRITICAL** (Target Design) |
| `12_gradient_boosting` | Ensemble modeling using XGBoost, LightGBM, CatBoost with time-series CV. | `phase14/models/` | Feature importance ranking, hyperparameter bounds, predictions logging. | Adapt XTrading's 50/50 XGBoost + CatBoost ensemble. | In-sample overfitting if CV uses random shuffle. | **CRITICAL** (Model Engine) |
| `16_strategy_simulation` | Decoupling model metrics (RMSE/AUC) from financial trading metrics (PF/Sharpe). | `phase14/validation/` | Model evaluation vs. Strategy simulation separation. | Evaluate RMSE/AUC in step 1, PF/Sharpe in step 2. | Assuming good RMSE guarantees positive PF. | **CRITICAL** (Evaluator Parity) |
| `18_transaction_costs` | Explicit friction modeling: exchange fees, slippage, spread, funding rates. | `phase14/execution/` | Multi-layer cost deduction (Taker fee + Slippage + Spread + Funding). | Implement Binance Futures taker fee (0.05% - 0.10%) + slippage + 8h funding. | Ignoring funding rates in perpetual swaps. | **CRITICAL** (Cost Modeling) |
| `19_risk_management` | Dynamic position sizing based on volatility, expected edge, and portfolio risk. | `phase14/risk/` & `phase14/portfolio/` | Volatility-adjusted sizing ($Position = \frac{Edge}{Volatility \times RiskDist}$). | Adapt ATR-based stop loss & leverage caps. | Over-leveraging on low-volatility regimes. | **HIGH** (Risk Management) |
| `20_strategy_synthesis` | Combining technical, ML, and regime signals into unified strategy execution. | `phase14/strategy/` | Multi-signal aggregation & regime routing. | Combine ML return predictions with regime filters. | Signal correlation leading to concentrated risk. | **HIGH** (Strategy Layer) |
| `25_live_trading` & `26_mlops` | Production verification, pipeline checks, circuit breakers, drift detection. | `experiments/watchdog.py` | Pipeline heartbeat, closed-candle verification, gap handling. | Integrate closed-candle gating and pipeline integrity. | Evaluating incomplete candles. | **HIGH** (Operational Safety) |
| `case_studies/crypto_perps` | Binance 1H OHLCV + 8H funding rate perpetual futures analysis. | `phase14/data/` | Perpetual swap data loading & funding rate alignment. | Process 1H OHLCV alongside Binance perpetual funding rates. | Timestamp mismatch between 1H candles and 8H funding. | **HIGH** (Crypto Perps Data) |

---

## 2. Key Code Patterns Extracted for Phase 14

### Pattern A: Decoupled Model vs. Strategy Evaluation
```
           STEP 1: MODEL EVALUATION                     STEP 2: STRATEGY EVALUATION
┌──────────────────────────────────────────────┐   ┌──────────────────────────────────────────────┐
│ Evaluate Directional Accuracy (Sign Agreement)│   │ Apply Pre-registered Hurdle (±T)             │
│ Evaluate Magnitude Correlation (|r̂| vs |r|)  ├──►│ Calculate Net Expected Edge after Costs      │
│ Evaluate Long & Short Tail Returns (Exp A/B) │   │ Measure Realized PF, Sharpe, Max DD          │
└──────────────────────────────────────────────┘   └──────────────────────────────────────────────┘
```

### Pattern B: Complete Cost Deduction Formula
$$\text{Net PnL}_{\text{trade}} = \text{Gross PnL} - (\text{Notional}_{\text{entry}} \times \text{Fee}_{\text{taker}}) - (\text{Notional}_{\text{exit}} \times \text{Fee}_{\text{taker}}) - (\text{Notional} \times \text{Slippage}) - \text{Funding}_{\text{accumulated}}$$

---

## 3. Structural Isolation & Firewall Guarantee

Phase 14 is built completely inside `phase14/` and `strategy_lab/`. No code in `phase14/` will import from or modify `paper_trading/`, `pipeline/`, or `experiments/check_live_progress.py`.
