import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VARMAX
from granger import plot_granger_results, plot_varma_fits, plot_varma_heatmap, granger_causality_test

def run_optimized_part4():
    # 1. 設置路徑 (請根據你的實際輸出目錄修改)
    output_dir = "E:/VSCODE/HKUST/MSDM5058/output/20260427_231201" 
    figure_dir = f"{output_dir}/figures"
    result_dir = f"{output_dir}/results"
    
    # 2. 讀取現有的數據
    # 假設數據已經由之前的 preprocess_data 生成
    ko_data = pd.read_csv(f"{output_dir}/data/KO_processed.csv")['Returns_Centered']
    jpm_data = pd.read_csv(f"{output_dir}/data/JPM_processed.csv")['Returns_Centered']
    
    ticker1, ticker2 = "KO", "JPM"
    data = pd.concat([ko_data, jpm_data], axis=1)
    data.columns = [ticker1, ticker2]
    
    print(f"--- [Part 4] Optimized Analysis for {ticker1} vs {ticker2} ---")

    # 3. 直接使用最優階數 AR=6, MA=0 進行擬合
    # 增加 maxiter 並更換 method 以提高收斂概率
    best_ar, best_ma = 6, 0
    print(f"[VARMA] Refitting the best AIC model (AR={best_ar}, MA={best_ma})...")
    
    model = VARMAX(data, order=(best_ar, best_ma), trend='c')
    try:
        # 使用 powell 算法通常比默認的 L-BFGS 更穩健，雖然慢一點
        result = model.fit(maxiter=500, disp=True, method='powell')
        print("[OK] Model converged successfully.")
    except Exception as e:
        print(f"[ERROR] Fit failed: {e}. Using the previous result or defaults.")
        # 如果還是失敗，可以嘗試默擬合，或減小 maxiter
        result = model.fit(maxiter=100, disp=False)

    # 4. 重新生成圖表
    
    # 圖 4.1: 直接從 CSV 讀取並畫熱力圖 (不需要重新跑搜索)
    order_table = pd.read_csv(f"{result_dir}/varma_order_table_{ticker1}_{ticker2}.csv")
    fig_heatmap = plot_varma_heatmap(order_table, ticker1, ticker2)
    fig_heatmap.savefig(f"{figure_dir}/varma_order_heatmap_AR6_{ticker1}_{ticker2}.png", dpi=300)
    print(f"[SAVED] Heatmap saved to {figure_dir}")

    # 圖 4.2: 格蘭傑因果檢驗 (這部分很快，可以重跑以確保一致性)
    gc_results = granger_causality_test(data, maxlag=10, name1=ticker1, name2=ticker2)
    fig_pvals = plot_granger_results(gc_results, ticker1, ticker2)
    fig_pvals.savefig(f"{figure_dir}/granger_pvalues_AR6_{ticker1}_{ticker2}.png", dpi=300)
    print(f"[SAVED] P-values plot saved to {figure_dir}")

    # 圖 4.3: VARMA(6,0) 擬合對比圖
    fig_fit = plot_varma_fits(data, result, ticker1, ticker2)
    fig_fit.savefig(f"{figure_dir}/varma_fit_AR6_{ticker1}_{ticker2}.png", dpi=300)
    print(f"[SAVED] VAR(6) Fit plot saved to {figure_dir}")

    # 5. 保存新的係數表
    coef_df = pd.DataFrame({
        'parameter': result.params.index,
        'value': result.params.values,
        'std_err': result.bse.values,
        'p_value': result.pvalues.values
    })
    # 添加顯著性標籤
    coef_df['significance'] = coef_df['p_value'].apply(
        lambda x: '***' if x < 0.001 else ('**' if x < 0.01 else ('*' if x < 0.05 else ''))
    )
    coef_df.to_csv(f"{result_dir}/varma_coefficients_AR6_{ticker1}_{ticker2}.csv", index=False)
    print(f"[DONE] Analysis completed. Check {result_dir} for new coefficients.")

if __name__ == "__main__":
    run_optimized_part4()