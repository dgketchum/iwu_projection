import json
import os
from itertools import product
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

from climate_indices import compute, indices

COLS = ['et', 'cc', 'ppt', 'eto', 'eff_ppt']

IDX_KWARGS = dict(distribution=indices.Distribution.gamma,
                  data_start_year=1985,
                  calibration_year_initial=1985,
                  calibration_year_final=2023,
                  periodicity=compute.Periodicity.monthly)


def correlations(npy_dir, out_dir, procs, metric, standardize_water_use):
    """Calculates and saves the correlation and linear regression coefficients
    between meteorological drought (SPI) and irrigation water use (IWU). For now,
    the only function to run a subsequent projection is calc='cc' and  standardize_water_use=False.

    TODO: needs a test/train split for validation.

    This function systematically explores the relationship between SPI and IWU for
    numerous agricultural fields. It iterates through various SPI timescales and
    calculates the IWU either as a standardized index or as a direct rolling
    sum, controlled by the `standardize_water_use` flag. For now, it only looks at the calendar
    year IWU and long meteorological time periods. These can be extended with a little modification.

    Using multiprocessing,
    it computes the Pearson's correlation, regression slope, intercept, and
    p-value for the relationship. All results are compiled and saved to a
    single CSV file for subsequent analysis.

    Args:
        npy_dir (str): Path to the directory containing the input .npy data file.
        out_dir (str): Path to the root directory for saving the output CSV file.
        procs (int): Number of parallel processes to use for computation.
        metric (str): The raw water use metric to use ('kc' for et/eto or 'cc').
        standardize_water_use (bool): Flag to determine IWU calculation. If True,
            computes a standardized index. If False, computes a rolling sum.

    Returns:
        str: The full path to the saved output CSV file.
    """

    hyd_areas = [(f.split('.')[0], os.path.join(npy_dir, f)) for f in os.listdir(npy_dir) if f.endswith('.npy')]
    hyd_areas = sorted(hyd_areas, key=lambda x: x[0])

    met_periods = [12, 18, 24, 30, 36]

    for hydro_area, npy_file in hyd_areas:

        print('\n', npy_file)
        js = npy_file.replace('.npy', '_index.json')
        input_array = np.load(npy_file)

        with open(js, 'r') as fp:
            index = json.load(fp)['index']

        print(len(index), 'fields')
        dt_range = [pd.to_datetime('{}-{}-01'.format(y, m)) for y in range(1980, 2025) for m in range(1, 13)]
        months = np.multiply(np.ones((len(index), len(dt_range))), np.array([dt.month for dt in dt_range]))
        series = {}
        for met_p in met_periods:

            if metric == 'kc':
                et_data = input_array[:, :, COLS.index('et')].copy() / input_array[:, :, COLS.index('eto')].copy()

            elif metric == 'cc':
                et_data = input_array[:, :, COLS.index('cc')].copy()
                et_data[et_data < 0.] = 0.

            elif metric == 'cu_frac':
                et_data = input_array[:, :, COLS.index('cc')].copy() / input_array[:, :, COLS.index('et')].copy()
                et_data[et_data < 0.] = 0.

            else:
                raise ValueError

            if standardize_water_use:
                iwu = np.apply_along_axis(lambda x: indices.spi(x, scale=12, **IDX_KWARGS), arr=et_data, axis=1)

            else:
                iwu = pd.DataFrame(data=np.array(et_data).T, index=dt_range, columns=index)
                iwu = iwu.rolling(window=12, min_periods=12, closed='right').sum()
                iwu = iwu.values.T

            ppt = input_array[:, :, COLS.index('ppt')].copy()

            spi = np.apply_along_axis(lambda x: indices.spi(x, scale=met_p, **IDX_KWARGS), arr=ppt, axis=1)

            stack = np.stack([iwu[:, -len(dt_range):], spi[:, -len(dt_range):], months])

            # looking back from the end of a growing season
            for from_month in range(12, 13):

                d_unmasked = stack[:, stack[2] == float(from_month)].copy().reshape((3, len(index), -1))
                mx = np.ma.masked_array(np.repeat(np.isnan(d_unmasked[:1, :, :]), 3, axis=0))
                d = np.ma.MaskedArray(d_unmasked, mx)

                a = ArrayDisAssembly(d)
                arrays = a.disassemble(n_sections=procs)

                if procs > 1:
                    with Pool(processes=procs) as p:
                        # Pearson's Correlation
                        pool_results_corr = [p.apply_async(pearsons_correlation, args=(a_,)) for a_ in arrays]
                        corefs = [res.get() for res in pool_results_corr]
                        corefs = np.array([item for sublist in corefs for item in sublist])

                        # Linear Regression and p-value
                        pool_results_reg = [p.apply_async(linear_regression, args=(a_,)) for a_ in arrays]
                        lin_reg_coeffs = [res.get() for res in pool_results_reg]
                        lin_reg_coeffs = np.array([item for sublist in lin_reg_coeffs for item in sublist])
                else:
                    corefs = np.array(pearsons_correlation(arrays[0]))
                    lin_reg_coeffs = np.array(linear_regression(arrays[0]))

                slope = lin_reg_coeffs[:, 0]
                intercept = lin_reg_coeffs[:, 1]
                p_value = lin_reg_coeffs[:, 2]

                col_corr = 'met{}_ag{}_fr{}_corr'.format(met_p, 12, from_month)
                series[col_corr] = corefs
                print(f'{col_corr:30} correlation: {np.nanmin(corefs):.4f} to {np.nanmax(corefs):.4f}')

                col_slope = 'met{}_ag{}_fr{}_slope'.format(met_p, 12, from_month)
                series[col_slope] = slope
                print(f'{col_slope:30} slope: {np.nanmin(slope):.4f} to {np.nanmax(slope):.4f}')

                col_intercept = 'met{}_ag{}_fr{}_intercept'.format(met_p, 12, from_month)
                series[col_intercept] = intercept
                print(f'{col_intercept:30} intercept: {np.nanmin(intercept):.4f} to {np.nanmax(intercept):.4f}')

                col_pvalue = 'met{}_ag{}_fr{}_pvalue'.format(met_p, 12, from_month)
                series[col_pvalue] = p_value
                print(f'{col_pvalue:30} p-value: {np.nanmin(p_value):.4f} to {np.nanmax(p_value):.4f}\n\n')

        cols = sorted(list(series.keys()))
        df_data = np.array([series[k] for k in cols]).T
        df = pd.DataFrame(index=index, data=df_data, columns=cols)

        if standardize_water_use:
            std_desc = 'standardized'
        else:
            std_desc = 'rolling_sum'

        odir =  os.path.join(out_dir, f'{metric}_Fof_SPI', std_desc)

        if not os.path.isdir(odir):
            os.makedirs(odir, exist_ok=True)

        ofile = os.path.join(odir, '{}.csv'.format(hydro_area))

        for k, v in series.items():
            df[k] = v

        df.to_csv(ofile)


def split(a, n):
    k, m = divmod(len(a), n)
    inds = [(i * k + min(i, m), (i + 1) * k + min(i + 1, m)) for i in range(n)]
    return inds


class ArrayDisAssembly(object):

    def __init__(self, arr):
        self.arrays = None
        self.n_sections = None
        self.assembled = None
        self.axis = None

        if isinstance(arr, list):
            self.arrays = arr
            self.assembled = self.assemble(arr)

        self.original = arr
        self.shape = arr.shape

    def disassemble(self, n_sections, axis=1):
        self.arrays = np.array_split(self.original, n_sections, axis=axis)
        self.n_sections = n_sections
        return self.arrays

    def assemble(self, results, axis=1):
        d = {r.idx: r.arr for r in results}
        l = [d[k] for k in sorted(d.keys())]
        self.assembled = np.concatenate(l, axis=axis)
        return self.assembled


def pearsons_correlation(d_):
    coref = [np.ma.corrcoef(d_[0, i, :], d_[1, i, :])[0][1].item() for i in range(d_.shape[1])]
    return coref


def linear_regression(d_):
    coeffs = []

    if np.ma.is_masked(d_):
        d_ = np.ma.filled(d_, fill_value=np.nan)

    for i in range(d_.shape[1]):
        x = d_[1, i, :]
        y = d_[0, i, :]

        valid_indices = ~np.isnan(x) & ~np.isnan(y)
        x_valid = x[valid_indices]
        y_valid = y[valid_indices]

        if x_valid.size < 2:
            coeffs.append([np.nan, np.nan])
            continue

        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_valid, y_valid)
            coeffs.append([slope, intercept, p_value])
        except (np.linalg.LinAlgError, ValueError):
            coeffs.append([np.nan, np.nan, np.nan])

    return coeffs


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    nv_data = os.path.join(root, 'Nevada', 'dri_field_pts')

    summaries = os.path.join(nv_data, 'fields_data', 'field_summaries')
    indir = os.path.join(nv_data, 'fields_data', 'fields_npy')
    odir_ = os.path.join(nv_data, 'fields_data', 'correlation_analysis')

    correlations(indir, odir_, procs=6, metric='cu_frac', standardize_water_use=False)

# ========================= EOF ====================================================================
