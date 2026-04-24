"""
MSDM5058 Financial Time Series Analysis Package
"""

from .preprocessing import download_data, compute_returns, preprocess_data
from .stationarity import adf_test, plot_acf_pacf, compare_arma
from .fractal import hurst_exponent, dfa_analysis, multifractal_analysis, mfdfa_analysis
from .granger import varma_model, granger_causality_test
from .fourier import fourier_transform, power_spectrum
from .emd import emd_decomposition, analyze_imfs
