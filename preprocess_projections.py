import os
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import pandas as pd
from tqdm import tqdm

REMAP_COLS = {'ETa_final_acre_ft': 'et', 'NetET_final_acre_ft': 'cc', 'PPT_MM': 'ppt',
              'ETO_MM': 'eto', 'Eff_PPT_Adjusted_acre_ft': 'eff_ppt'}

COLS = ['et', 'cc', 'ppt', 'eto', 'eff_ppt']

FUTURE_SCENARIO_LIST = ['rcp45', 'rcp85']

MODEL_LIST = ['bcc-csm1-1',
              'bcc-csm1-1-m',
              'BNU-ESM',
              'CanESM2',
              'CCSM4',
              'CNRM-CM5',
              'CSIRO-Mk3-6-0',
              'GFDL-ESM2G',
              'GFDL-ESM2M',
              'HadGEM2-CC365',
              'HadGEM2-ES365',
              'inmcm4',
              'IPSL-CM5A-MR',
              'IPSL-CM5A-LR',
              'IPSL-CM5B-LR',
              'MIROC5',
              'MIROC-ESM',
              'MIROC-ESM-CHEM',
              'MRI-CGCM3',
              'NorESM1-M']

GRIDMET_RESAMPLE_MAP = {'year': 'first',
                        'month': 'first',
                        'day': 'first',
                        'centroid_lat': 'first',
                        'centroid_lon': 'first',
                        'elev_m': 'first',
                        'eto_mm': 'sum',
                        'prcp_mm': 'sum',
                        'eto_mm_uncorr': 'sum'}


def split_projections(fields, raw_exports, outdir, num_workers=None, gridmet_correction=None):
    """Restructures raw, year-based climate projections into continuous, field-specific time series files.

    I needlessly downloaded daily projection data, so it should be modified to run on monthly when another
    extract is made using Chris Pearson's maca_loop.py.

    This function processes a large collection of raw, downscaled MACA V2 climate
    projection files in daily timesteps, which are organized as one file per
    year, model, and scenario. It reads these files in parallel, aggregates
    daily data to monthly sums, optionally applies a bias correction to ETo,
    and then reorganizes the entire dataset. The final output consists of one
    compressed Parquet file per spatial identifier (GFID), where each file

    Args:
        fields (str): Path to a CSV file containing a list of unique 'GFID's
            to process.
        raw_exports (str): Path to the directory containing the raw,
            year-by-year projection CSV files.
        outdir (str): Path to the output directory where the final
            GFID-specific Parquet files will be saved.
        num_workers (int, optional): The number of parallel worker processes
            to use for reading and writing files. Defaults to None, which
            may use all available CPU cores depending on the executor.
        gridmet_correction (str, optional): Path to a JSON file containing
            monthly ETo bias-correction factors, keyed by GFID.
    """

    fields = pd.read_csv(fields)
    fields['GFID'] = fields['GFID'].astype(str)
    gfids = fields['GFID'].unique()

    if gridmet_correction:
        with open(gridmet_correction, 'r') as f_obj:
            corrections = json.load(f_obj)
        corrections = {gfid: {int(kk): vv for kk, vv in v.items() if kk in [str(e) for e in range(1, 13)]} for gfid, v
                       in
                       corrections.items()}
    else:
        corrections = None

    gfid_data_collector = defaultdict(lambda: defaultdict(list))

    proj_files = []
    for model in MODEL_LIST:
        for scenario in FUTURE_SCENARIO_LIST:

            add_files = []

            for yr in range(2025, 2100):
                file_ = os.path.join(raw_exports, f'{scenario}_{model}_{yr}.csv')
                if not os.path.exists(file_):
                    print(f'{os.path.basename(file_)} does not exist')
                    break

                add_files.append(file_)

            proj_files.extend(add_files)

    if num_workers == 1:
        results = []
        for proj_file in proj_files:
            result = _read_and_process_proj_file(proj_file, correction_dict=corrections)
            results.append(result)

    else:

        process_func = partial(_read_and_process_proj_file, correction_dict=corrections)

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(executor.map(process_func, proj_files),
                                total=len(proj_files),
                                desc="Reading projection files"))

    for file_result in results:
        if file_result is None:
            print(f'no files were read')
            continue
        for gfid, data in file_result.items():
            for (scenario, model), df_list in data.items():
                gfid_data_collector[gfid][(scenario, model)].extend(df_list)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    tasks = []
    for gfid in gfids:
        if gfid in gfid_data_collector:
            tasks.append((gfid, gfid_data_collector[gfid], outdir))

    if num_workers == 1:
        for task in tasks:
            _process_and_write_gfid(task)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            list(tqdm(executor.map(_process_and_write_gfid, tasks),
                      total=len(tasks),
                      desc='Processing and writing projections'))


def _read_and_process_proj_file(proj_file, correction_dict=None):
    if not os.path.exists(proj_file):
        return None

    basename = os.path.basename(proj_file).replace('.csv', '')
    parts = basename.split('_')
    scenario = parts[0]
    model = '_'.join(parts[1:-1])

    file_data_collector = defaultdict(lambda: defaultdict(list))

    try:
        df = pd.read_csv(proj_file, usecols=['GFID', 'datenum', 'pr', 'eto_mm'], engine='c')
        df['date'] = pd.to_datetime(df['datenum'], format='%Y%m%d')
        df['GFID'] = df['GFID'].astype(str)
        df_monthly = df.groupby('GFID').resample('MS', on='date')[['pr', 'eto_mm']].sum()

        if correction_dict is not None:
            monthly_correction_dicts = df_monthly.index.get_level_values('GFID').map(correction_dict)
            months = df_monthly.index.get_level_values('date').month
            correction_factors = [
                monthly_dict[month]['eto']
                for monthly_dict, month in zip(monthly_correction_dicts, months)
            ]
            df_monthly['eto_corrected'] = df_monthly['eto_mm'] * correction_factors

        for gfid, group_df in df_monthly.groupby(level='GFID'):
            file_data_collector[gfid][(scenario, model)].append(group_df.droplevel(0))

        return dict(file_data_collector)
    except Exception as e:
        print(f"Error processing file {proj_file}: {e}")
        return None


def _process_and_write_gfid(args):
    gfid, gfid_data, outdir = args

    processed_projections = []
    for (scenario, model), df_list in gfid_data.items():
        full_ts_df = pd.concat(df_list, axis=0)

        full_ts_df.rename(columns={'pr': f'{scenario}_{model}_ppt',
                                   'eto_mm': f'{scenario}_{model}_eto',
                                   'eto_corrected': f'{scenario}_{model}_eto_corrected',
                                   }, inplace=True)

        processed_projections.append(full_ts_df)

    if not processed_projections:
        return

    final_df = pd.DataFrame(index=pd.to_datetime([]))
    for df in processed_projections:
        final_df = final_df.join(df, how='outer')

    out_file = os.path.join(outdir, f'{gfid}.parquet.gz')
    final_df.to_parquet(out_file, compression='gzip')


if __name__ == '__main__':

    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    nv_data = os.path.join(root, 'Nevada', 'dri_field_pts')

    fields_data = os.path.join(nv_data, 'fields_data')

    npy_dir = os.path.join(fields_data, 'fields_npy')
    fields_gis = os.path.join(nv_data, 'fields_gis')
    nv_fields_boundaries = os.path.join(fields_gis, 'Nevada_Agricultural_Field_Boundaries_20250214')
    gfid_fields = os.path.join(nv_fields_boundaries,
                               'Nevada_Agricultural_Field_Boundaries_20250214_5071_GFID.csv')

    gridmet_factors = os.path.join(nv_fields_boundaries, 'Nevada_Fields_with_Nearest_GFID.json')

    projections_extracts_ = os.path.join(fields_data, 'projections', 'exports')
    projections_processed_ = os.path.join(fields_data, 'projections', 'processed')
    met = os.path.join(fields_data, 'gridmet')
    split_projections(gfid_fields, projections_extracts_, projections_processed_,
                      gridmet_correction=gridmet_factors, num_workers=12)

# ========================= EOF ====================================================================
