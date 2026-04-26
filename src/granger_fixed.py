import os
import time
import warnings
import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VARMAX
from statsmodels.tsa.stattools import grangercausalitytests


def prepare_bivariate_series(x1, x2, name1='X1', name2='X2'):
    s1 = pd.Series(x1).reset_index(drop=True)
    s2 = pd.Series(x2).reset_index(drop=True)
    df = pd.concat([s1, s2], axis=1)
    df.columns = [name1, name2]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def _fit_varma_worker(data, ar, ma, maxiter, queue):
    try:
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            model = VARMAX(data, order=(ar, ma), trend='c', enforce_stationarity=True)
            result = model.fit(disp=False, maxiter=maxiter)

        warning_msgs = [str(w.message) for w in wlist]

        queue.put({
            'ok': True,
            'ar': ar,
            'ma': ma,
            'aic': result.aic,
            'bic': result.bic,
            'warnings': warning_msgs,
        })
    except Exception as e:
        queue.put({
            'ok': False,
            'ar': ar,
            'ma': ma,
            'error': repr(e),
        })


def select_varma_order(
    data,
    max_ar=4,
    max_ma=4,
    maxiter=50,
    timeout_sec=300,
    verbose=True
):
    best_aic = np.inf
    best_bic = np.inf
    best_aic_order = None
    best_bic_order = None
    rows = []

    total = (max_ar + 1) * (max_ma + 1) - 1
    count = 0
    global_start = time.time()

    if verbose:
        print(f"\n[VARMA] Start order search")
        print(f"[VARMA] data shape = {data.shape}")
        print(f"[VARMA] total combinations = {total}")
        print(f"[VARMA] timeout per fit = {timeout_sec}s")
        print(f"[VARMA] maxiter per fit = {maxiter}")

    for ar in range(max_ar + 1):
        for ma in range(max_ma + 1):
            if ar == 0 and ma == 0:
                continue

            count += 1
            start = time.time()

            if verbose:
                print(f"\n[TRY] ({count}/{total}) VARMA({ar},{ma}) start ...", flush=True)

            queue = mp.Queue()
            proc = mp.Process(
                target=_fit_varma_worker,
                args=(data, ar, ma, maxiter, queue)
            )
            proc.start()
            proc.join(timeout=timeout_sec)

            elapsed = time.time() - start

            if proc.is_alive():
                proc.terminate()
                proc.join()

                msg = f"VARMA({ar},{ma}) exceeded {timeout_sec}s and was skipped."
                warnings.warn(msg)

                if verbose:
                    print(f"[TIMEOUT] {msg} elapsed={elapsed:.2f}s", flush=True)

                rows.append({
                    'ar': ar,
                    'ma': ma,
                    'aic': np.nan,
                    'bic': np.nan,
                    'status': 'timeout',
                    'elapsed_sec': elapsed,
                    'warning': msg
                })
                continue

            if queue.empty():
                msg = f"VARMA({ar},{ma}) ended with no returned result."
                warnings.warn(msg)

                if verbose:
                    print(f"[FAIL] {msg} elapsed={elapsed:.2f}s", flush=True)

                rows.append({
                    'ar': ar,
                    'ma': ma,
                    'aic': np.nan,
                    'bic': np.nan,
                    'status': 'empty',
                    'elapsed_sec': elapsed,
                    'warning': msg
                })
                continue

            out = queue.get()

            if out['ok']:
                warning_text = " | ".join(out.get('warnings', []))
                if verbose:
                    print(
                        f"[OK ] VARMA({ar},{ma}) finished in {elapsed:.2f}s | "
                        f"AIC={out['aic']:.4f}, BIC={out['bic']:.4f}",
                        flush=True
                    )
                    if warning_text:
                        print(f"[WARN] VARMA({ar},{ma}): {warning_text}", flush=True)

                rows.append({
                    'ar': ar,
                    'ma': ma,
                    'aic': out['aic'],
                    'bic': out['bic'],
                    'status': 'ok',
                    'elapsed_sec': elapsed,
                    'warning': warning_text
                })

                if out['aic'] < best_aic:
                    best_aic = out['aic']
                    best_aic_order = (ar, ma)

                if out['bic'] < best_bic:
                    best_bic = out['bic']
                    best_bic_order = (ar, ma)
            else:
                msg = out['error']
                warnings.warn(f"VARMA({ar},{ma}) failed: {msg}")

                if verbose:
                    print(f"[FAIL] VARMA({ar},{ma}) failed after {elapsed:.2f}s | {msg}", flush=True)

                rows.append({
                    'ar': ar,
                    'ma': ma,
                    'aic': np.nan,
                    'bic': np.nan,
                    'status': 'failed',
                    'elapsed_sec': elapsed,
                    'warning': msg
                })

    total_elapsed = time.time() - global_start
    order_table = pd.DataFrame(rows)

    ok_table = order_table[order_table['status'] == 'ok'].copy()
    if ok_table.empty:
        raise RuntimeError('VARMA order selection failed for all candidate orders')

    ok_table = ok_table.sort_values(['bic', 'aic']).reset_index(drop=True)

    if verbose:
        print(f"\n[VARMA] Order search finished in {total_elapsed:.2f}s")
        print(f"[VARMA] Best AIC order = {best_aic_order}, AIC = {best_aic:.4f}")
        print(f"[VARMA] Best BIC order = {best_bic_order}, BIC = {best_bic:.4f}")

    return best_aic_order, best_bic_order, order_table


def varma_model(data, order=(1, 0), maxiter=100, verbose=True):
    if verbose:
        print(f"\n[VARMA] Fitting final VARMA{order} ...", flush=True)

    start = time.time()
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        model = VARMAX(data, order=order, trend='c', enforce_stationarity=True)
        result = model.fit(disp=False, maxiter=maxiter)

    elapsed = time.time() - start

    if verbose:
        print(f"[VARMA] Final VARMA{order} finished in {elapsed:.2f}s", flush=True)
        if wlist:
            for w in wlist:
                print(f"[WARN] Final VARMA{order}: {w.message}", flush=True)

    return result


def get_coefficients(result):
    coef_table = pd.DataFrame({
        'parameter': result.params.index,
        'value': result.params.values,
        'std_err': result.bse.values,
        'p_value': result.pvalues.values,
    })
    coef_table['significance'] = np.select(
        [coef_table['p_value'] < 0.01, coef_table['p_value'] < 0.05, coef_table['p_value'] < 0.1],
        ['***', '**', '*'],
        default=''
    )
    return coef_table


def granger_causality_test(data, maxlag=10, name1='X1', name2='X2', verbose=True):
    if verbose:
        print(f"\n[GC] Running Granger causality tests, maxlag={maxlag} ...", flush=True)

    start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        test_1 = grangercausalitytests(data[[name1, name2]], maxlag=maxlag, verbose=False)
        test_2 = grangercausalitytests(data[[name2, name1]], maxlag=maxlag, verbose=False)

    pvalues_21 = [test_1[lag][0]['ssr_ftest'][1] for lag in range(1, maxlag + 1)]
    pvalues_12 = [test_2[lag][0]['ssr_ftest'][1] for lag in range(1, maxlag + 1)]

    elapsed = time.time() - start

    results = {
        f'{name2}_causes_{name1}': {
            'pvalues': pvalues_21,
            'significant_lags': [i + 1 for i, p in enumerate(pvalues_21) if p < 0.05]
        },
        f'{name1}_causes_{name2}': {
            'pvalues': pvalues_12,
            'significant_lags': [i + 1 for i, p in enumerate(pvalues_12) if p < 0.05]
        }
    }

    if verbose:
        print(f"[GC] Finished in {elapsed:.2f}s", flush=True)
        print(f"[GC] {name2} -> {name1} significant lags: {results[f'{name2}_causes_{name1}']['significant_lags']}")
        print(f"[GC] {name1} -> {name2} significant lags: {results[f'{name1}_causes_{name2}']['significant_lags']}")

    return results


def plot_granger_results(results, name1='X1', name2='X2', figsize=(12, 5)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, key, title, color in [
        (axes[0], f'{name2}_causes_{name1}', f'{name2} → {name1}', 'steelblue'),
        (axes[1], f'{name1}_causes_{name2}', f'{name1} → {name2}', 'darkorange')
    ]:
        pvals = np.maximum(np.asarray(results[key]['pvalues'], dtype=float), 1e-300)
        lags = np.arange(1, len(pvals) + 1)
        ax.bar(lags, pvals, color=color, alpha=0.8)
        ax.axhline(0.05, color='red', ls='--', label='p=0.05')
        ax.set_yscale('log')
        ax.set_xlabel('Lag')
        ax.set_ylabel('F-test p-value')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_varma_fits(data, result, ticker1, ticker2, figsize=(12, 6)):
    fitted = result.fittedvalues
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    axes[0].plot(data.index, data[ticker1], color='lightcoral', alpha=0.25, label='data')
    axes[0].plot(data.index, fitted[ticker1], color='firebrick', lw=1.2, label=f'VARMA{result.model.order}')
    axes[0].set_title(ticker1)
    axes[0].set_ylabel(f'return {ticker1}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(data.index, data[ticker2], color='mediumpurple', alpha=0.25, label='data')
    axes[1].plot(data.index, fitted[ticker2], color='navy', lw=1.2, label=f'VARMA{result.model.order}')
    axes[1].set_title(ticker2)
    axes[1].set_ylabel(f'return {ticker2}')
    axes[1].set_xlabel('time t')
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    plt.tight_layout()
    return fig


def plot_varma_heatmap(order_table, ticker1, ticker2, figsize=(7, 6), top_n=10):
    ok_table = order_table[order_table['status'] == 'ok'].copy()
    ok_table = ok_table.sort_values('aic', ascending=True).reset_index(drop=True)
    ok_table['rank'] = np.arange(1, len(ok_table) + 1)

    pivot = ok_table.pivot(index='ar', columns='ma', values='aic')

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(pivot.values, cmap='turbo', aspect='auto', origin='lower')
    ax.grid(False)
    ax.set_facecolor('white')

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_xlabel('MA(q)')
    ax.set_ylabel('AR(p)')
    ax.set_title(f'{ticker1}\n{ticker2}')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('AIC')

    top_orders = ok_table[ok_table['rank'] <= top_n]
    for _, row in top_orders.iterrows():
        ar = row['ar']
        ma = row['ma']
        rank = int(row['rank'])

        if ar in pivot.index and ma in pivot.columns:
            y = list(pivot.index).index(ar)
            x = list(pivot.columns).index(ma)
            ax.text(x, y, str(rank), ha='center', va='center', color='white', fontsize=10, fontweight='bold')

    best_row = ok_table.iloc[0]
    best_ar, best_ma = best_row['ar'], best_row['ma']
    y_best = list(pivot.index).index(best_ar)
    x_best = list(pivot.columns).index(best_ma)
    rect = plt.Rectangle((x_best - 0.5, y_best - 0.5), 1, 1, fill=False, edgecolor='black', lw=2.5)
    ax.add_patch(rect)

    plt.tight_layout()
    return fig


def analyze_granger(
    returns1,
    returns2,
    ticker1,
    ticker2,
    output_dir,
    max_ar=4,
    max_ma=4,
    max_lag=10,
    selection='bic',
    search_maxiter=50,
    final_maxiter=100,
    timeout_sec=300,
    verbose=True
):
    os.makedirs(f'{output_dir}/figures', exist_ok=True)
    os.makedirs(f'{output_dir}/results', exist_ok=True)

    total_start = time.time()

    if verbose:
        print(f"\n[Part 4] Granger Causality Analysis: {ticker1} vs {ticker2}")
        print(f"[Part 4] Preparing bivariate series ...")

    data = prepare_bivariate_series(returns1, returns2, ticker1, ticker2)

    if verbose:
        print(f"[Part 4] Bivariate data shape = {data.shape}")

    best_aic, best_bic, order_table = select_varma_order(
        data,
        max_ar=max_ar,
        max_ma=max_ma,
        maxiter=search_maxiter,
        timeout_sec=timeout_sec,
        verbose=verbose
    )

    selected_order = best_aic if selection.lower() == 'aic' else best_bic

    if verbose:
        print(f"\n[Part 4] Selected order by {selection.upper()} = {selected_order}")
        print(f"[Part 4] AIC-best = {best_aic}, BIC-best = {best_bic}")

    result = varma_model(
        data,
        order=selected_order,
        maxiter=final_maxiter,
        verbose=verbose
    )

    coef_table = get_coefficients(result)
    gc_results = granger_causality_test(
        data,
        maxlag=max_lag,
        name1=ticker1,
        name2=ticker2,
        verbose=verbose
    )

    fig1 = plot_granger_results(gc_results, ticker1, ticker2)
    fig1.savefig(f'{output_dir}/figures/granger_pvalues_{ticker1}_{ticker2}.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)

    fig2 = plot_varma_fits(data, result, ticker1, ticker2)
    fig2.savefig(f'{output_dir}/figures/varma_fit_{ticker1}_{ticker2}.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)

    fig3 = plot_varma_heatmap(order_table, ticker1, ticker2)
    fig3.savefig(f'{output_dir}/figures/varma_order_heatmap_{ticker1}_{ticker2}.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)

    order_table.to_csv(f'{output_dir}/results/varma_order_table_{ticker1}_{ticker2}.csv', index=False)
    coef_table.to_csv(f'{output_dir}/results/varma_coefficients_{ticker1}_{ticker2}.csv', index=False)

    total_elapsed = time.time() - total_start

    if verbose:
        print(f"\n[Part 4] Done in {total_elapsed:.2f}s")
        print(f"[Part 4] Results saved to: {output_dir}")

    summary = {
        'best_aic_order': best_aic,
        'best_bic_order': best_bic,
        'selected_order': selected_order,
        'granger_results': gc_results,
        'coef_table': coef_table,
        'order_table': order_table,
        'data': data,
        'model_result': result,
        'elapsed_sec': total_elapsed,
    }
    return summary