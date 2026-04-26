# MSDM5058 Project I - Financial Time Series Analysis

This project implements a comprehensive financial time series analysis following the requirements of MSDM5058 Project I. It analyzes stock market data using advanced econometric and fractal analysis techniques.

## 📊 Project Overview

The project analyzes daily stock price data for AAPL and MSFT from 2008 to 2024, implementing all six required parts:

1. **Data Preprocessing** - Download and process stock data
2. **Stationarity Analysis** - ADF tests, ACF/PACF, ARMA modeling
3. **Fractal Analysis** - Hurst exponent, DFA, Multifractal analysis
4. **Granger Causality** - VARMA models and causality testing
5. **Fourier Analysis** - FFT, Power Spectral Density, Spectrograms
6. **Empirical Mode Decomposition** - EMD with IMF analysis

## 🚀 Quick Start

### Installation

1. Clone or download the project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

Run the complete analysis:
```bash
python run_analysis.py
```

This will:
- Download AAPL and MSFT data (if not already available)
- Execute all 6 parts of the analysis
- Generate plots and save results to organized directories
- Create an HTML report

### Using Individual Modules

```python
# Import modules
from src.preprocessing import download_data, compute_returns
from src.stationarity import adf_test, plot_acf_pacf
from src.fractal import hurst_exponent, dfa_analysis
from src.granger import varma_model, granger_causality_test
from src.fourier import fourier_transform, power_spectrum
from src.emd import emd_decomposition

# Example: Download and preprocess data
data = download_data(['AAPL'], '2008-01-02', '2024-04-01')
returns = compute_returns(data['AAPL'])
```

## 📁 Project Structure

```
MSDM5058 Project I/
├── src/                    # Source code modules
│   ├── __init__.py        # Package initialization
│   ├── preprocessing.py    # Part 1: Data preprocessing
│   ├── stationarity.py    # Part 2: Stationarity analysis
│   ├── fractal.py         # Part 3: Fractal analysis
│   ├── granger.py         # Part 4: Granger causality
│   ├── fourier.py         # Part 5: Fourier analysis
│   └── emd.py            # Part 6: EMD analysis
├── data/                  # Stock data (CSV files)
│   ├── AAPL.csv
│   └── MSFT.csv
├── output/                # Analysis results (generated)
│   ├── [timestamp]/
│   │   ├── figures/       # All plots
│   │   ├── results/       # CSV results
│   │   └── data/         # Processed data
├── config.py              # Configuration settings
├── report_generator.py    # HTML report generation
├── run_analysis.py        # Main script
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## 🔧 Configuration

Edit `config.py` to customize:
- Date ranges for analysis
- Output directories
- Plot parameters
- Analysis settings

## 📈 Analysis Results

The analysis generates:

### Visualizations
- Price and return series
- ACF/PACF plots
- Hurst exponent analysis
- DFA plots
- Multifractal spectra
- FFT and power spectral density
- EMD decomposition with IMFs
- Comparison charts

### Statistical Outputs
- ADF test results
- ARMA model selection
- Granger causality p-values
- Fractal dimensions
- Spectral analysis results
- Summary statistics

### Reports
- Interactive HTML report with embedded plots
- CSV files for all results
- Summary comparison tables

## 📊 Interpretation Guide

### Hurst Exponent
- H > 0.5: Persistent (trending)
- H ≈ 0.5: Random walk
- H < 0.5: Anti-persistent (mean-reverting)

### DFA α
- α > 1: Non-stationary (persistent)
- α = 1: Brownian motion
- 0.5 < α < 1: Stationary with memory
- α < 0.5: Anti-persistent

### Multifractal Width
- Width > 0.1: Multifractal (complex dynamics)
- Width < 0.1: Monofractal (simple dynamics)

## 🛠️ Dependencies

### Core
- numpy, pandas, matplotlib, seaborn
- scipy, statsmodels

### Optional (enhanced features)
- nolds: Advanced fractal analysis
- MFDFA: Multifractal DFA
- PyEMD: Alternative EMD implementation

## 📋 Project Requirements Compliance

| Requirement | Status | Module |
|-------------|---------|--------|
| 6+ years daily data | ✅ | preprocessing.py |
| Log returns calculation | ✅ | preprocessing.py |
| ADF test | ✅ | stationarity.py |
| ACF/PACF | ✅ | stationarity.py |
| ARMA modeling | ✅ | stationarity.py |
| Hurst exponent | ✅ | fractal.py |
| DFA | ✅ | fractal.py |
| Multifractal | ✅ | fractal.py |
| Granger causality | ✅ | granger.py |
| FFT analysis | ✅ | fourier.py |
| Power spectral density | ✅ | fourier.py |
| EMD decomposition | ✅ | emd.py |
| IMF analysis | ✅ | emd.py |

## 🔍 Troubleshooting

### Common Issues

1. **Data Download Fails**
   - Check internet connection
   - yfinance may need updated authentication

2. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Memory Issues**
   - Reduce dataset size in config.py
   - Close other applications

4. **Plot Display Issues**
   - Use non-interactive backend: `matplotlib.use('Agg')`

## 📝 Notes

- The analysis uses ~4000 daily observations per stock
- All statistical tests are at 95% confidence level
- EMD is implemented in pure Python for portability
- HTML reports are self-contained with embedded plots

## 🎯 Next Steps

1. Extend to more stocks
2. Add real-time data feed
3. Implement portfolio analysis
4. Add machine learning components
5. Create interactive dashboard

## 📚 References

- Mandelbrot, B. (1997). The Fractal Geometry of Nature
- Peng, C.K. et al. (1994). Mosaic organization of DNA nucleotides
- H.E. Hurst (1951). Long-term storage capacity of reservoirs