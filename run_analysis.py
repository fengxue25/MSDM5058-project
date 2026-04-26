import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("src")
from config import AnalysisConfig
from src.preprocessing import download_data, preprocess_data, plot_prices_and_returns
from src.stationarity import (
    compare_arma_guess_with_best,
    arma_order_heatmap,
    adf_test,
    plot_acf_pacf,
    plot_acf_absolute,
    compare_arma,
    compare_linear_nonlinear_acf,
)
from src.fractal import (
    hurst_exponent,
    plot_hurst,
    dfa_analysis,
    plot_dfa,
    multifractal_analysis,
    plot_multifractal,
    mfdfa_analysis,
    plot_mfdfa,
)
from src.granger_fixed import analyze_granger
from src.fourier import (
    fourier_transform,
    power_spectrum,
    plot_fourier,
    plot_power_spectrum,
    analyze_spectral_properties,
)
from src.emd import (
    emd_decomposition,
    analyze_imfs,
    plot_selected_imfs,
    plot_hurst_imf,
    analyze_imf_psd,
    analyze_reduced_series,
)

plt.style.use(AnalysisConfig.PLOT_STYLE)
plt.rcParams.update(AnalysisConfig.CUSTOM_RC)
sns.set_palette(AnalysisConfig.COLOR_PALETTE["primary"])


def save_results_to_csv(results, output_dir, filename):
    pd.DataFrame([results]).to_csv(
        os.path.join(output_dir, "results", filename), index=False
    )


def analyze_single_stock(ticker, ticker_name, prices, returns_centered, output_dir):
    print(f"\n[Part 2] {ticker} Stationarity Analysis...")
    adf_result = adf_test(returns_centered, ticker)
    fig1 = plot_acf_pacf(returns_centered, lags=40, title=ticker_name)
    fig1.savefig(
        f"{output_dir}/figures/{ticker}_acf_pacf.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig1)
    fig2 = plot_acf_absolute(returns_centered, lags=40, title=ticker_name)
    fig2.savefig(
        f"{output_dir}/figures/{ticker}_acf_absolute.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig2)
    guess_order = (1, 1)
    compare_arma(returns_centered, guess_order)

    # ARMA order selection heatmap
    fig_arma, arma_table = arma_order_heatmap(
        returns_centered, max_ar=9, max_ma=9, criterion="aic", title=ticker_name
    )
    fig_arma.savefig(
        f"{output_dir}/figures/{ticker}_arma_order_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig_arma)

    # 新增：猜测 vs 最优
    arma_guess_check = compare_arma_guess_with_best(guess_order, arma_table)

    print(f"\nGuessed ARMA order from ACF/PACF: {arma_guess_check['guess_order']}")
    print(f"Best ARMA order by AIC: {arma_guess_check['best_order']}")

    if arma_guess_check["guess_rank"] is not None:
        print(f"Guessed order rank: {arma_guess_check['guess_rank']}")
        print(f"Guessed order AIC: {arma_guess_check['guess_criterion']:.4f}")

    print(f"Best AIC: {arma_guess_check['best_criterion']:.4f}")

    if arma_guess_check["match"]:
        print("The guessed order matches the best fit.")
    else:
        print("The guessed order does not match the best fit.")

    compare_linear_nonlinear_acf(returns_centered, lags=40)

    print(f"\n[Part 3] {ticker} Fractal Analysis...")
    hurst_result = hurst_exponent(returns_centered)
    fig_hurst = plot_hurst(hurst_result, title=ticker_name)
    fig_hurst.savefig(
        f"{output_dir}/figures/{ticker}_hurst.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig_hurst)

    dfa_result = dfa_analysis(returns_centered)
    dfa_alpha = dfa_result["alpha"]
    fig_dfa = plot_dfa(dfa_result, title=ticker_name)
    fig_dfa.savefig(
        f"{output_dir}/figures/{ticker}_dfa.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig_dfa)

    q_values = np.array([1.0, 2.0, 3.0, 4.0])
    multifractal_result = multifractal_analysis(
        prices.dropna().values, q_values=q_values
    )
    fig_mf = plot_multifractal(multifractal_result, title=ticker_name)
    fig_mf.savefig(
        f"{output_dir}/figures/{ticker}_multifractal.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig_mf)

    mfdfa_result = mfdfa_analysis(np.asarray(returns_centered), q_values=q_values)
    fig_mfdfa = plot_mfdfa(mfdfa_result, title=ticker_name)
    fig_mfdfa.savefig(
        f"{output_dir}/figures/{ticker}_mfdfa.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig_mfdfa)

    print(f"\n[Part 5] {ticker} Fourier Analysis...")
    fft_result = fourier_transform(returns_centered)
    psd_result = power_spectrum(returns_centered)
    fig_fft = plot_fourier(fft_result, title=ticker_name)
    fig_fft.savefig(
        f"{output_dir}/figures/{ticker}_fourier_magnitude.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig_fft)
    fig_psd = plot_power_spectrum(psd_result, title=ticker_name)
    fig_psd.savefig(
        f"{output_dir}/figures/{ticker}_power_spectrum.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig_psd)
    fourier_results = analyze_spectral_properties(psd_result, title=ticker_name)

    print(f"\n[Part 6] {ticker} EMD Analysis...")
    x = (
        returns_centered.values
        if hasattr(returns_centered, "values")
        else np.asarray(returns_centered)
    )
    emd_results = emd_decomposition(x, max_imf=10)
    emd_info = analyze_imfs(emd_results)  # 這裡包含各個 IMF 的 Hurst
    k = emd_results["n_imfs"]
    selected_indices = [1, max(1, k // 4), max(1, k // 2), max(1, (3 * k) // 4), k]
    fig_sel = plot_selected_imfs(
        emd_results, indices=selected_indices, title=ticker_name
    )
    fig_sel.savefig(
        f"{output_dir}/figures/{ticker}_emd_selected_imfs.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig_sel)

    imf_analysis = analyze_imfs(emd_results)
    fig_hurst_imf = plot_hurst_imf(imf_analysis["hurst_values"], title=ticker_name)
    fig_hurst_imf.savefig(
        f"{output_dir}/figures/{ticker}_imf_hurst.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig_hurst_imf)

    if emd_results["n_imfs"] >= 2:
        fig_psd_imf = analyze_imf_psd(
            emd_results["imfs"], title=ticker_name, indices=[0, 1]
        )
        fig_psd_imf.savefig(
            f"{output_dir}/figures/{ticker}_imf1_imf2_psd.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig_psd_imf)

    fig_reduced = analyze_reduced_series(emd_results, title=ticker_name)
    fig_reduced.savefig(
        f"{output_dir}/figures/{ticker}_reduced_series_psd.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig_reduced)

    results = {
        "ticker": ticker,
        "n_obs": len(prices),
        "adf_statistic": adf_result["adf_statistic"],
        "adf_pvalue": adf_result["p_value"],
        "is_stationary": adf_result["is_stationary"],
        # "hurst": hurst_result["H"],
        # "dfa_alpha": dfa_result["alpha"],
    }
    save_results_to_csv(results, output_dir, f"{ticker}_results.csv")
    export_summary_stats(
        ticker, fourier_results, emd_info, hurst_result, dfa_alpha, output_dir
    )
    return results


def export_summary_stats(
    ticker, fourier_results, emd_results, hurst_val, dfa_alpha, output_dir
):
    """
    將散落在各個步驟的關鍵數值匯總並保存，方便寫報告使用
    """
    summary_data = []

    # 1. 保存 Fourier 關鍵數據 (Spectral Slope 和 Peak Frequency)
    summary_data.append(
        {
            "Category": "Fourier",
            "Metric": "Spectral Slope (Beta)",
            "Value": fourier_results.get("spectral_slope", "N/A"),
        }
    )
    if len(fourier_results.get("peak_frequencies", [])) > 0:
        summary_data.append(
            {
                "Category": "Fourier",
                "Metric": "Primary Peak Frequency",
                "Value": fourier_results["peak_frequencies"][0],
            }
        )

    # 2. 保存基礎分形數據
    summary_data.append(
        {"Category": "Fractal", "Metric": "Hurst Exponent (R/S)", "Value": hurst_val}
    )
    summary_data.append(
        {"Category": "Fractal", "Metric": "DFA Alpha", "Value": dfa_alpha}
    )

    # 3. 保存 EMD 的各個 IMF Hurst 指數
    imf_hursts = emd_results.get("hurst_values", [])
    for i, h in enumerate(imf_hursts):
        summary_data.append(
            {"Category": "EMD_IMF_Hurst", "Metric": f"IMF_{i+1}_Hurst", "Value": h}
        )

    # 轉化為 DataFrame 並導出
    df_summary = pd.DataFrame(summary_data)
    save_path = f"{output_dir}/results/{ticker}_summary_report_data.csv"
    df_summary.to_csv(save_path, index=False)
    print(f">>> Summary data for {ticker} saved to {save_path}")


def main():
    print("MSDM5058 Project I - Financial Time Series Analysis")
    directories = AnalysisConfig.get_directories()
    output_dir = directories[0]
    # output_dir = os.path.dirname(directories[0])
    print(f"output_dir : {output_dir}")

    os.makedirs(f"{output_dir}/data", exist_ok=True)
    os.makedirs(f"{output_dir}/figures", exist_ok=True)
    os.makedirs(f"{output_dir}/results", exist_ok=True)

    ticker_map = {"KO": "The Coca-Cola Company", "JPM": "JPMorgan Chase & Co."}
    tickers = list(ticker_map.keys())

    start_date = "2008-01-02"
    end_date = "2024-04-01"
    data = download_data(tickers, start_date, end_date, f"{output_dir}/data")

    stocks_data = {}
    all_results = []
    for ticker, ticker_name in ticker_map.items():
        print(f"\nTicker: {ticker}, Name: {ticker_name}")
        prices = data[ticker]
        _, returns, returns_centered = preprocess_data(prices, ticker, output_dir)
        stocks_data[ticker] = returns_centered
        fig = plot_prices_and_returns(
            prices, returns_centered, ticker_name, figsize=(12, 8)
        )
        fig.savefig(
            f"{output_dir}/figures/{ticker}_price_centeredreturns.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
        results = analyze_single_stock(
            ticker, ticker_name, prices, returns_centered, output_dir
        )
        all_results.append(results)

    # print("\n[Part 4] Granger Causality Analysis...")
    # analyze_granger(
    #     [tickers[0]],
    #     stocks_data[tickers[1]],
    #     tickers[0],
    #     tickers[1],
    #     output_dir,
    #     max_ar=9,
    #     max_ma=9,
    #     max_lag=10,
    # )
    # print(f"\nAnalysis complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
