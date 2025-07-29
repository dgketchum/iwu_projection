import json
import os
import numpy as np
import pandas as pd
from scipy import stats
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
                   from_month=12, metric='cc'):
    """Projects future irrigation water use (IWU) based on historical models. For each agricultural field,
    this function identifies the best-performing historical regression model from the correlation analysis CSV build
    in analysis.py. Best performing in this context only means highest correlation, positive or negative. It then
    applies this model to future precipitation projections (SPI) to predict future IWU. The projections for all
    climate models and scenarios are saved to a separate CSV file for each field.

    Args:
        correlations_csv_dir (str): Path to the CSV with correlation and
            regression coefficient results from a previous analysis.
        historical_npy_dir (str): Directory containing historical .npy data.
        future_parquet_dir (str): Directory containing future projections data in .parquet format
        out_dir (str): The directory where output projection files will be saved.
        gfid_csv (str): Path to a CSV file mapping field IDs to grid IDs (GFID).
        from_month (int, optional): The month of the year for which the
            analysis is performed. Defaults to 12.
        metric (str): choice of water use proxy; cc: 'crop consumption' or netET, kc: crop coefficient (ETa/ETo),
                        'cu_frac': netET/ETa, 'cu_eto': netET/ETo
        target_areas (list): selection of hydrographic area codes to limit analysis to

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

            best_models[field_id] = {
                'met_p': met_p,
                'slope': correlations_df.loc[field_id, slope_col],
                'intercept': correlations_df.loc[field_id, intercept_col]
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
                future_field_df = pd.read_parquet(field_parquet_path)
            except (KeyError, FileNotFoundError):
                continue

            field_projections = []
            model_params = best_models[field_id]
            met_p = model_params['met_p']

            first = True

            for model_name in MODEL_LIST:
                for scenario in FUTURE_SCENARIO_LIST:

                    future_col_name = f'{scenario}_{model_name}'

                    ppt_col_name = f'{future_col_name}_ppt'
                    eto_col_name = f'{future_col_name}_eto'

                    if ppt_col_name not in future_field_df.columns:
                        continue

                    future_ppt_series = future_field_df[ppt_col_name]
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

                        historical_netet = historical_data[i, :, COLS.index('cc')]
                        historical_netet = pd.Series(historical_netet, index=hist_dt_range)
                        historical_netet.name = 'historical_netET'
                        field_projections.append(historical_netet)

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
                    future_eto_series.name = f'{model_name}_{scenario}_eto'
                    field_projections.append(future_eto_series)

                    if weighted:
                        predicted_cc = preditced_et_metric * future_eto_annual
                    else:
                        predicted_cc = preditced_et_metric

                    predicted_cc.name = f'{model_name}_{scenario}_netET'
                    field_projections.append(predicted_cc)

                    s_spi.name = f'{model_name}_{scenario}_spi'
                    field_projections.append(s_spi.loc[future_dt_range])

                    future_ppt_series.name = f'{model_name}_{scenario}_ppt'
                    field_projections.append(future_ppt_series)

            if not field_projections:
                continue

            projection_df = pd.concat(field_projections, axis=1)

            output_filename = os.path.join(out_dir, f'projected_{metric}_{field_id}.parquet')
            projection_df.to_csv(output_filename)

            print(f"{output_filename} {projection_df.shape}")


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    nv_data_dir = os.path.join(root, 'Nevada', 'dri_field_pts')
    historical_npy_dir_ = os.path.join(nv_data_dir, 'fields_data', 'fields_npy')
    results_dir = os.path.join(nv_data_dir, 'fields_data', 'correlation_analysis')

    calculation_types = ['kc', 'cc', 'cu_eto']

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
        )

# ========================= EOF ====================================================================
