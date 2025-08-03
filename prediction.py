import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from climate_indices import compute, indices

COLS = ['et', 'cc', 'ppt', 'eto', 'eff_ppt']

FUTURE_SCENARIO_LIST = ['rcp45', 'rcp85']

MODEL_LIST = ['bcc-csm1-1', 'bcc-csm1-1-m', 'BNU-ESM', 'CanESM2', 'CCSM4',
              'CNRM-CM5', 'CSIRO-Mk3-6-0', 'GFDL-ESM2G', 'GFDL-ESM2M',
              'HadGEM2-CC365', 'HadGEM2-ES365', 'inmcm4', 'IPSL-CM5A-MR',
              'IPSL-CM5A-LR', 'IPSL-CM5B-LR', 'MIROC5', 'MIROC-ESM',
              'MIROC-ESM-CHEM', 'MRI-CGCM3', 'NorESM1-M']


def project_net_et(correlations_csv_dir, historical_npy_dir,
                   future_parquet_dir, out_dir, gfid_csv,
                   target_areas=None, weighted=False,
                   from_month=12, metric='cc', plot_dir=None):
    """Projects future water use for fields using historical drought-response models.

        For each agricultural field, this function applies a previously developed statistical
        model to project future agricultural water use (e.g., net crop consumptive use).
        The process is as follows:

        1.  For a given field, it identifies the "best" historical linear regression model
            from the correlation analysis results. The "best" model is defined as the one
            with the highest absolute correlation between a historical water use metric
            and the Standardized Precipitation Index (SPI) at a specific timescale.
        2.  It loads the future climate projections (precipitation and ETo) for that
            field's location, which cover multiple climate models (GCMs) and scenarios (RCPs).
        3.  For each GCM/scenario, it calculates a continuous SPI time series across the
            combined historical and future period, ensuring the SPI is calibrated only on
            the historical data.
        4.  It applies the field's best-fit linear model (y = slope * x + intercept) to the
            future SPI values (x) to predict the future annual water use metric (y).
        5.  It saves a comprehensive Parquet file for the field, containing the historical
            data, all future climate driver data (PPT, ETo, SPI), and the final projected
            water use for every model and scenario.

        Args:
            correlations_csv_dir (str): Path to the directory containing the CSV files with
                correlation and regression coefficients from the analysis step.
            historical_npy_dir (str): Path to the directory containing the historical
                .npy data arrays and their corresponding .json index files.
            future_parquet_dir (str): Path to the directory containing the processed future
                climate projections as location-specific .parquet files.
            out_dir (str): The root directory where output projection files will be saved.
            gfid_csv (str): Path to a CSV file that maps field OPENET_IDs to their
                corresponding location GFIDs.
            target_areas (list, optional): A list of hydrographic area codes to process.
                If None, all available areas will be processed. Defaults to None.
            weighted (bool, optional): If True, the predicted water use metric (which is
                assumed to be a ratio like 'kc') is multiplied by the annual future ETo
                to yield an absolute consumptive use value. Defaults to False.
            from_month (int, optional): The end-of-season month used to extract annual
                values from the time series (e.g., 10 for October). Defaults to 12.
            metric (str, optional): The water use metric being projected, used for naming
                output files. Defaults to 'cc'.

        Returns:
            None. The function writes projection results to Parquet files.
        """

    hyd_areas = [(f.split('.')[0], os.path.join(historical_npy_dir, f)) for f in
                 os.listdir(historical_npy_dir) if f.endswith('.npy')]
    hyd_areas = sorted(hyd_areas, key=lambda x: x[0])

    for hydro_area, npy_file in hyd_areas:

        if target_areas and hydro_area not in target_areas:
            continue

        corr_file = os.path.join(correlations_csv_dir, f'{hydro_area}.csv')

        correlations_df = pd.read_csv(corr_file, index_col=0)

        fields_gridmap = pd.read_csv(gfid_csv, index_col='OPENET_ID')
        fields_gridmap = {i: r['GFID'] for i, r in fields_gridmap.iterrows()}

        corr_cols = [c for c in correlations_df.columns if '_corr' in c]
        s_best_corr_col = correlations_df[corr_cols].abs().idxmax(axis=1)

        best_models = {}
        for field_id, best_corr_col in s_best_corr_col.items():
            parts = best_corr_col.split('_')
            met_p = int(parts[0].replace('met', ''))

            slope_col = best_corr_col.replace('_corr', '_slope')
            intercept_col = best_corr_col.replace('_corr', '_intercept')
            pval_col = best_corr_col.replace('_corr', '_pvalue')
            corr_col = best_corr_col.replace('_corr', '_corr')

            best_models[field_id] = {
                'met_p': met_p,
                'slope': correlations_df.loc[field_id, slope_col],
                'correlation': correlations_df.loc[field_id, corr_col],
                'intercept': correlations_df.loc[field_id, intercept_col],
                'pvalue': correlations_df.loc[field_id, pval_col]
            }

        historical_data = np.load(npy_file)

        with open(npy_file.replace('.npy', '_index.json'), 'r') as fp:
            field_index = json.load(fp)['index']

        hist_dt_range = pd.to_datetime([f'{y}-{m}-01' for y in range(1980, 2025) for m in range(1, 13)])

        for i, field_id in enumerate(field_index):

            if field_id not in best_models:
                continue

            try:
                field_gfid = fields_gridmap[field_id]
                field_parquet_path = os.path.join(future_parquet_dir, f'{field_gfid}.parquet.gz')
                if not os.path.exists(field_parquet_path):
                    continue
                # to get rolling data in 2025 we need an extra year
                future_field_df = pd.read_parquet(field_parquet_path)
                # spi calculation is on joined time series and must not overlap
                future_field_df_trunc = future_field_df.loc[
                    [i for i in future_field_df.index if i not in hist_dt_range]]
            except (KeyError, FileNotFoundError):
                continue

            field_projections = []
            model_params = best_models[field_id]
            met_p = model_params['met_p']

            first = True

            for model_name in MODEL_LIST:
                for scenario in FUTURE_SCENARIO_LIST:

                    # if model_name != 'HadGEM2-CC365':
                    #     continue

                    future_col_name = f'{scenario}_{model_name}'

                    ppt_col_name = f'{future_col_name}_ppt'
                    eto_col_name = f'{future_col_name}_eto_corrected'

                    if ppt_col_name not in future_field_df.columns:
                        raise ValueError

                    future_ppt_series = future_field_df_trunc[ppt_col_name]
                    future_ppt = future_ppt_series.values
                    future_dt_range = future_ppt_series.index
                    full_dt_range = hist_dt_range.union(future_dt_range)

                    historical_ppt = historical_data[i, :, COLS.index('ppt')]

                    if first:
                        historical_ppt = pd.Series(historical_ppt, index=hist_dt_range)
                        historical_ppt.name = 'historical_ppt'
                        field_projections.append(historical_ppt)

                        historical_eto = historical_data[i, :, COLS.index('eto')]
                        historical_eto = pd.Series(historical_eto, index=hist_dt_range)
                        historical_eto.name = 'historical_eto'
                        field_projections.append(historical_eto)

                        if metric == 'kc':
                            historical_iwu = historical_data[i, :, COLS.index('et')]
                            historical_iwu = pd.Series(historical_iwu, index=hist_dt_range)
                            historical_iwu.name = 'historical_ETa'

                        else:
                            historical_iwu = historical_data[i, :, COLS.index('cc')]
                            historical_iwu = pd.Series(historical_iwu, index=hist_dt_range)
                            historical_iwu.name = 'historical_netET'

                        historical_iwu = pd.Series(historical_iwu, index=hist_dt_range)
                        field_projections.append(historical_iwu)

                        historical_netet_rolling = historical_iwu.rolling(window=12, min_periods=12,
                                                                          closed='right').sum()
                        historical_netet_wye = historical_netet_rolling[
                            historical_netet_rolling.index.month == from_month]
                        historical_netet_wye = pd.Series(historical_netet_wye, index=hist_dt_range)

                        if metric == 'kc':
                            historical_netet_wye.name = 'historical_eta_wye'
                        else:
                            historical_netet_wye.name = 'historical_netet_wye'

                        field_projections.append(historical_netet_wye)

                        first = False

                    full_ppt = np.concatenate([historical_ppt, future_ppt])

                    spi = indices.spi(full_ppt, scale=met_p, distribution=indices.Distribution.gamma,
                                      data_start_year=hist_dt_range.year[0],
                                      calibration_year_initial=hist_dt_range.year[0],
                                      calibration_year_final=hist_dt_range.year[-1],
                                      periodicity=compute.Periodicity.monthly)

                    s_spi = pd.Series(spi, index=full_dt_range)
                    future_spi_for_month = s_spi[s_spi.index.month == from_month].loc[future_dt_range[0]:]

                    preditced_et_metric = model_params['slope'] * future_spi_for_month + model_params['intercept']

                    future_eto_series = future_field_df[eto_col_name]
                    eto_rolling = future_eto_series.rolling(window=12, min_periods=12, closed='right').sum()
                    future_eto_annual = eto_rolling[eto_rolling.index.month == from_month].loc[future_dt_range[0]:]
                    future_eto_series.name = f'{model_name}_{scenario}_eto_corrected'
                    field_projections.append(future_eto_series)

                    if weighted:
                        predicted_iwu = preditced_et_metric * future_eto_annual
                    else:
                        predicted_iwu = preditced_et_metric

                    if metric == 'kc':
                        predicted_iwu.name = f'{model_name}_{scenario}_ETa'
                    else:
                        predicted_iwu.name = f'{model_name}_{scenario}_netET'

                    field_projections.append(predicted_iwu)

                    s_spi.name = f'{model_name}_{scenario}_spi'
                    field_projections.append(s_spi.loc[future_dt_range])

                    future_ppt_series.name = f'{model_name}_{scenario}_ppt'
                    field_projections.append(future_ppt_series)

            if not field_projections:
                continue

            projection_df = pd.concat(field_projections, axis=1)

            output_filename = os.path.join(out_dir, f'projected_{metric}_{field_id}.parquet')

            if plot_dir:
                print(f'\n\n{metric} {field_id}')
                print(f'{field_id} historical water year end IWU: {historical_netet_wye.mean()}')
                print(f'{field_id} projection correlation: {model_params["correlation"]}')
                print(f'{field_id} projection p-value: {model_params["pvalue"]}')

                rcp45 = np.nanmean(projection_df[[c for c in projection_df.columns if 'rcp45_netET' in c]].values)
                print(f'{field_id} projected mean RCP 4.5 netET: {rcp45}')
                rcp85 = np.nanmean(projection_df[[c for c in projection_df.columns if 'rcp85_netET' in c]].values)
                print(f'{field_id} projected mean RCP 8.5 netET: {rcp85}')

                plot_water_use_projection(projection_df, field_id, metric, out_dir=plot_dir)

            projection_df.to_parquet(output_filename)

            print(f"{output_filename} {projection_df.shape}")


def plot_water_use_projection(projection_df, field_id, metric, out_dir):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    if metric == 'kc':
        hist_col = 'historical_eta_wye'
        future_col_suffix = '_ETa'
    else:
        hist_col = 'historical_netet_wye'
        future_col_suffix = '_netET'

    historical_data = projection_df[[hist_col]].dropna()
    ax.plot(historical_data.index, historical_data[hist_col],
            label='Historical', color='black', linewidth=2)

    colors = {'rcp45': 'dodgerblue', 'rcp85': 'firebrick'}
    for scenario in FUTURE_SCENARIO_LIST:
        future_cols = [c for c in projection_df.columns if f'{scenario}{future_col_suffix}' in c]
        future_df = projection_df[future_cols].dropna(how='all')

        mean_projection = future_df.mean(axis=1)
        std_projection = future_df.std(axis=1)
        upper_bound = mean_projection + std_projection
        lower_bound = mean_projection - std_projection

        ax.plot(mean_projection.index, mean_projection,
                label=f'Mean {scenario.upper()}', color=colors[scenario], linestyle='--')

        ax.fill_between(future_df.index, lower_bound, upper_bound,
                        color=colors[scenario], alpha=0.2, label=f'{scenario.upper()} Range (±1σ)')

    ax.set_title(f'{metric.upper()}-based Historical and Projected Water Use for Field: {field_id}', fontsize=16)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel(f'Annual Water Use ({metric.upper()})', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    plot_filename = os.path.join(out_dir, f'projection_plot_{metric}_{field_id}.png')
    plt.savefig(plot_filename, dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    nv_data_dir = os.path.join(root, 'Nevada', 'dri_field_pts')
    historical_npy_dir_ = os.path.join(nv_data_dir, 'fields_data', 'fields_npy')
    results_dir = os.path.join(nv_data_dir, 'fields_data', 'correlation_analysis')

    calculation_types = ['cc', 'kc', 'cu_eto']

    for calculation_type in calculation_types:

        if calculation_type == 'cc':
            weighted_ = False
        else:
            weighted_ = True

        standardize_water_use = False

        if standardize_water_use:
            std_desc = 'standardized'
        else:
            std_desc = 'rolling_mean'

        correlations_csv_ = os.path.join(results_dir, f'{calculation_type}_Fof_SPI', std_desc)

        future_data_dir_ = os.path.join(nv_data_dir, 'fields_data', 'maca', 'processed')

        fields_gis = os.path.join(nv_data_dir, 'fields_gis')
        nv_fields_boundaries = os.path.join(fields_gis, 'Nevada_Agricultural_Field_Boundaries_20250214')
        gfid_fields_ = os.path.join(nv_fields_boundaries,
                                    'Nevada_Agricultural_Field_Boundaries_20250214_5071_GFID.csv')

        projection_out_dir_ = os.path.join(nv_data_dir, 'fields_data', 'iwu_projections', calculation_type)
        os.makedirs(projection_out_dir_, exist_ok=True)

        projection_plot_dir_ = os.path.join(nv_data_dir, 'fields_data', 'iwu_projection_plots', calculation_type)
        os.makedirs(projection_plot_dir_, exist_ok=True)

        project_net_et(
            correlations_csv_dir=correlations_csv_,
            historical_npy_dir=historical_npy_dir_,
            future_parquet_dir=future_data_dir_,
            from_month=10,
            weighted=weighted_,
            metric=calculation_type,
            out_dir=projection_out_dir_,
            gfid_csv=gfid_fields_,
            target_areas='117',
            plot_dir=None,
        )

# ========================= EOF ====================================================================
