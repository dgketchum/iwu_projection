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


def project_net_et(historical_desc, correlations_csv_path, historical_npy_dir,
                   future_parquet_dir, out_dir, gfid_csv, from_month=12):
    """Projects future irrigation water use (IWU) based on historical models.

    For each agricultural field, this function identifies the best-performing historical regression model from the
    correlation analysis CSV build in analysis.py. It then applies this model to future precipitation projections (
    SPI) to predict future IWU. The projections for all climate models and scenarios are saved to a separate CSV file
    for each field.

    Args:
        historical_desc (str): Identifier for historical .npy data files.
        correlations_csv_path (str): Path to the CSV with correlation and
            regression coefficient results from a previous analysis.
        historical_npy_dir (str): Directory containing historical .npy data.
        calculation (str): The water use metric used, e.g., 'cc' or 'simi'.
        future_parquet_dir (str): Directory of field-specific .parquet files
            containing future precipitation projections.
        out_dir (str): The directory where output projection files will be saved.
        gfid_csv (str): Path to a CSV file mapping field IDs to grid IDs (GFID).
        from_month (int, optional): The month of the year for which the
            analysis is performed. Defaults to 12.
        ag_scale (int, optional): The rolling window timescale in months for
            the agricultural data. Defaults to 12.
    """

    correlations_df = pd.read_csv(correlations_csv_path, index_col=0)

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

    historical_npy_path = os.path.join(historical_npy_dir, f'{historical_desc}.npy')
    historical_data = np.load(historical_npy_path)

    with open(historical_npy_path.replace('.npy', '_index.json'), 'r') as fp:
        field_index = json.load(fp)['index']

    hist_dt_range = pd.to_datetime([f'{y}-{m}-01' for y in range(1980, 2025) for m in range(1, 13)])

    for i, field_id in enumerate(field_index):
        print(f'Processing field {field_id}: {i + 1} of {len(field_index)}')

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

        for model_name in MODEL_LIST:
            for scenario in FUTURE_SCENARIO_LIST:
                future_col_name = f'{scenario}_{model_name}'
                ppt_col_name = f'{future_col_name}_ppt'

                if ppt_col_name not in future_field_df.columns:
                    continue

                historical_ppt = historical_data[i, :, COLS.index('ppt')]
                future_ppt_series = future_field_df[ppt_col_name]
                future_ppt = future_ppt_series.values
                future_dt_range = future_ppt_series.index

                full_ppt = np.concatenate([historical_ppt, future_ppt])
                full_dt_range = hist_dt_range.union(future_dt_range)

                spi = indices.spi(full_ppt, scale=met_p, distribution=indices.Distribution.gamma,
                                  data_start_year=hist_dt_range.year[0],
                                  calibration_year_initial=hist_dt_range.year[0],
                                  calibration_year_final=hist_dt_range.year[-1],
                                  periodicity=compute.Periodicity.monthly)

                s_spi = pd.Series(spi, index=full_dt_range)
                future_spi_for_month = s_spi[s_spi.index.month == from_month].loc[future_dt_range[0]:]

                projected_iwu = model_params['slope'] * future_spi_for_month + model_params['intercept']
                projected_iwu.name = f'{model_name}_{scenario}'
                field_projections.append(projected_iwu)

        if not field_projections:
            continue

        projection_df = pd.concat(field_projections, axis=1)
        projection_df.index = projection_df.index.year
        projection_df.index.name = 'Year'

        output_filename = os.path.join(out_dir, f'projected_iwu_{field_id}.csv')
        projection_df.to_csv(output_filename)

    print("Processing complete.")


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    nv_data_dir = os.path.join(root, 'Nevada', 'dri_field_pts')
    historical_npy_dir = os.path.join(nv_data_dir, 'fields_data', 'fields_npy')
    results_dir = os.path.join(nv_data_dir, 'fields_data', 'indices')

    calculation_type = 'cc'
    standardize_water_use = False
    if standardize_water_use:
        std_desc = 'standardized'
    else:
        std_desc = 'rolling_sum'

    correlations_csv = os.path.join(results_dir, calculation_type, '{}_{}.csv'.format(std_desc, calculation_type))

    future_data_dir = os.path.join(nv_data_dir, 'fields_data', 'projections', 'processed')

    fields_gis = os.path.join(nv_data_dir, 'fields_gis')
    nv_fields_boundaries = os.path.join(fields_gis, 'Nevada_Agricultural_Field_Boundaries_20250214')
    gfid_fields = os.path.join(nv_fields_boundaries,
                               'Nevada_Agricultural_Field_Boundaries_20250214_5071_GFID.csv')

    projection_out_dir = os.path.join(results_dir, 'projections')
    os.makedirs(projection_out_dir, exist_ok=True)

    project_net_et(
        historical_desc='field_summaries_EToF_final',
        correlations_csv_path=correlations_csv,
        historical_npy_dir=historical_npy_dir,
        future_parquet_dir=future_data_dir,
        out_dir=projection_out_dir,
        calculation=calculation_type,
        gfid_csv=gfid_fields)

# ========================= EOF ====================================================================
