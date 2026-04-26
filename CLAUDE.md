# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Architecture

This is a financial time series analysis project implementing MSDM5058 Project I requirements. The architecture follows a modular design where each analysis part is implemented in a separate module under `src/`.

### Core Structure

- **Main Entry Point**: `run_analysis.py` - Orchestrates the complete analysis workflow
- **Configuration**: `config.py` - Centralized settings using class-based configuration
- **Report Generation**: `report_generator.py` - Creates HTML reports with embedded figures
- **Data Storage**: `data/` - Contains AAPL.csv and MSFT.csv stock data
- **Output**: `output/[timestamp]/` - Organized results with figures/, results/, and data/ subdirectories

### Module Dependencies

Each module in `src/` is independent but follows a consistent interface:
- All plotting functions return matplotlib Figure objects
- Functions handle missing optional dependencies gracefully (e.g., nolds, MFDFA)
- EMD implementation is pure Python for portability
- Data flow: preprocessing → analysis → visualization → reporting

## Common Commands

### Running the Analysis
```bash
# Run complete analysis for both stocks
python run_analysis.py

# Run individual module (for testing)
python src/preprocessing.py
python src/stationarity.py
# ... etc for other modules
```

### Testing Individual Components
```python
# Test preprocessing
from src.preprocessing import download_data, compute_returns
data = download_data(['AAPL'], '2008-01-02', '2024-04-01')
returns = compute_returns(data['AAPL'])

# Test stationarity
from src.stationarity import adf_test, plot_acf_pacf
adf_result = adf_test(returns)
```

### Output Management
```python
# Use configuration for file paths
from config import AnalysisConfig
output_dir = AnalysisConfig.get_output_directory()
figure_path = AnalysisConfig.get_figure_path('test.png')
```

## Key Implementation Details

### EMD Algorithm
- Pure Python implementation in `emd.py` (no external EMD libraries required)
- Uses cubic interpolation for envelope generation
- IMF selection via `plot_selected_imfs()` follows project requirements exactly
- Hurst exponent calculated for each IMF with fallback methods

### Fractal Analysis
- Dual implementations for Hurst exponent: R/S analysis with optional nolds enhancement
- DFA supports polynomial detrending of any order
- Multifractal analysis includes both absolute moment method and MFDFA (if available)
- Automatic width calculation for multifractal characterization

### Statistical Analysis
- Granger causality uses VARMA framework with automatic order selection
- ADF tests include trend options via MethodConfig
- ARMA modeling uses AIC/BIC criteria with configurable max p/q values
- All p-values use 95% confidence level threshold

## Configuration Approach

The project uses a class-based configuration system:
- `AnalysisConfig`: Analysis parameters and paths
- `MethodConfig`: Method-specific settings
- `ReportConfig`: Report generation settings
- `DebugConfig`: Debugging and logging

### Adding New Analysis
1. Create new module in `src/` following existing pattern
2. Add functions to `src/__init__.py` for module-level access
3. Update `config.py` with any new parameters
4. Add analysis to `run_analysis.py` main workflow

### Data Pipeline
1. Download via `download_data()` (yfinance backend)
2. Compute log returns: `X = ln(S(t)/S(t-1))`
3. Remove mean: `Y = X - μ`
4. Process through analysis pipeline
5. Save results to timestamped output directory

## Special Considerations

### Performance
- EMD is O(k·n²) for k iterations and n points
- Large datasets (>10k points) may require parameter adjustments in config.py
- Consider memory usage when running multiple analyses

### Data Quality
- Automatically checks minimum 4000 observations requirement
- Handles missing data in preprocessing
- ADF tests include stationarity determination

### Visualization Standards
- All figures use consistent styling from AnalysisConfig
- Plots include grid lines, proper labels, and legends
- DPI set to 300 for publication quality
- Color scheme maintained across all visualizations

### Error Handling
- Graceful degradation when optional libraries unavailable
- Boundary handling in EMD interpolation
- Validation of input data lengths
- Comprehensive logging in DebugConfig