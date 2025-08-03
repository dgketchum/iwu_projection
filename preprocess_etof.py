import os
import json
import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm

REMAP_COLS = {'ETA_MM': 'et', 'ETNET_MM': 'cc', 'PPT_MM': 'ppt',
              'ETO_MM': 'eto', 'EFF_PPT_MM': 'eff_ppt'}

COLS = ['et', 'cc', 'ppt', 'eto', 'eff_ppt']

GRIDMET_RESAMPLE_MAP = {'year': 'first',
                        'month': 'first',
                        'day': 'first',
                        'centroid_lat': 'first',
                        'centroid_lon': 'first',
                        'elev_m': 'first',
                        'eto_mm': 'sum',
                        'prcp_mm': 'sum',
                        'eto_mm_uncorr': 'sum'}


def preprocess_historical(in_pqt, gridmet, gridmet_gfid, outdir, target_areas=None, overwrite=False,
                          anomalous_recs_file=None, expected_recs=480):

    """Processes and merges historical water use data with GridMET climate data.

        This function iterates through hydrographic areas and their associated agricultural
        fields. For each field, it loads the historical consumptive use data (from ET Demands)
        and the corresponding daily GridMET climate data.

        The core process involves:
        1. Resampling the daily GridMET data to a monthly time step.
        2. Aligning the field's consumptive use data to the same monthly index.
        3. Replacing the precipitation and reference ET in the field data with the
           standardized values from GridMET.
        4. Stacking the final time series for all fields within a hydrographic area
           into a 3D NumPy array.

        The output for each hydrographic area consists of two files: a .npy file
        containing the numerical data array and a .json file that provides an index
        mapping the array's first dimension to the specific field IDs.

        Args:
            in_pqt (str): Path to the directory containing input parquet files, where each
                file represents a hydrographic area and contains ET Demands model output.
            gridmet (str): Path to the directory containing daily GridMET climate data as
                CSV files, named by their GridMET ID (e.g., 'gridmet_12345.csv').
            gridmet_gfid (str): Path to the CSV file that maps field OPENET_IDs to
                their corresponding GridMET cell GFIDs.
            outdir (str): Path to the directory where the output .npy and .json index
                files will be saved.
            target_areas (list of str, optional): A list of specific hydrographic area IDs
                to process. If None, all areas in `in_pqt` will be processed. Defaults to None.
            overwrite (bool, optional): If True, existing output files will be overwritten.
                Defaults to False.
            anomalous_recs_file (str, optional): Path to save a JSON file logging fields
                that do not have the expected number of records. Defaults to None.
            expected_recs (int, optional): The expected number of monthly records in a
                complete time series (e.g., 40 years * 12 months = 480). Defaults to 480.

        Returns:
            None. The function saves files to disk.
        """

    fields = pd.read_csv(gridmet_gfid, index_col='OPENET_ID')

    hyd_areas = [(f.split('_')[0], os.path.join(in_pqt, f)) for f in os.listdir(in_pqt) if f.endswith('.parquet')]
    hyd_areas = sorted(hyd_areas, key=lambda x: x[0])

    misshape_records = {}

    for hydro_area, etof_file in hyd_areas:

        if target_areas and hydro_area not in target_areas:
            continue

        out_json = os.path.join(outdir, f'{hydro_area}_index.json')
        out_npy = os.path.join(outdir, f'{hydro_area}.npy')

        if os.path.exists(out_json) and os.path.exists(out_npy) and not overwrite:
            continue

        df = pd.read_parquet(etof_file)
        df.index = pd.DatetimeIndex(df['DATE'])
        zone_fids = list(set(df['OPENET_ID'].values))
        zone_fields = fields.loc[[i for i in fields.index if i in zone_fids]]

        # seems I added a field-GFID feature for all 'nearest' gridmet centroids, pick the first
        if len(zone_fields) != len(zone_fids):
            duplicated_indices = zone_fields.index.duplicated(keep='first')
            zone_fields = zone_fields[~duplicated_indices]

        first, idxes, array = True, [], None

        for i, (fid, v) in enumerate(tqdm(zone_fields.iterrows(),
                                          desc=f'Processing {hydro_area}',
                                          total=zone_fields.shape[0])):

            if fid != 'NV_20975':
                continue

            g_fid = str(int(v['GFID']))

            file_ = os.path.join(gridmet, 'gridmet_{}.csv'.format(g_fid))

            met_df = pd.read_csv(file_, index_col='date', parse_dates=True)
            met_new_cols = {c: f'{c}_gm' for c in met_df.columns}
            met_df = met_df.resample('MS').agg(GRIDMET_RESAMPLE_MAP)
            met_df = met_df.rename(columns=met_new_cols)

            subarray = df[df['OPENET_ID'] == fid].copy()
            subarray = subarray.rename(columns=REMAP_COLS)[COLS]

            subarray.loc[subarray['cc'] < 0, 'cc'] = 0.0
            subarray.loc[np.isnan(subarray['et']), 'et'] = 0.0

            if not len(subarray) == expected_recs:
                misshape_records[fid] = len(subarray)

            try:
                subarray = subarray.reindex(met_df.index)
            except ValueError:
                print(f'\nWarning: found {subarray.shape[0]} records in {fid}')
                duplicated_indices = subarray.index.duplicated(keep='first')
                subarray = subarray[~duplicated_indices]
                subarray = subarray.reindex(met_df.index)

            subarray['eto'] = met_df['eto_mm_gm'].copy()
            subarray['ppt'] = met_df['prcp_mm_gm'].copy()

            if first:
                array = np.zeros((len(zone_fids), len(subarray.index), len(REMAP_COLS))) * np.nan
                first = False

            idxes.append(fid)
            array[i, :, :] = subarray.values.reshape((1, len(subarray.index), len(REMAP_COLS)))

        with open(out_json, 'w') as f:
            json.dump({'index': idxes}, f, indent=4)

        np.save(out_npy, array)
        print(f'Hydro-Area {hydro_area}: saved {out_json}, len {len(idxes)}')
        print(f'Hydro-Area {hydro_area}: saved {out_npy}, shape: {array.shape}')

    with open(anomalous_recs_file, 'w') as f:
        json.dump(misshape_records, f, indent=4)

if __name__ == '__main__':

    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    nv_data = os.path.join(root, 'Nevada', 'dri_field_pts')

    fields_data = os.path.join(nv_data, 'fields_data')
    pqt_ = os.path.join(fields_data, 'NV_field_summaries_EToF_final_large.parquet')
    js_ = os.path.join(fields_data, 'NV_field_summaries_EToF_tiles.json')
    pqt_dir = os.path.join(fields_data, 'field_summaries')

    fields_gis = os.path.join(nv_data, 'fields_gis')
    nv_fields_boundaries = os.path.join(fields_gis, 'Nevada_Agricultural_Field_Boundaries_20250214')
    gridmet_factors_ = os.path.join(nv_fields_boundaries,
                                    'Nevada_Agricultural_Field_Boundaries_20250214_5071_GFID.csv')

    npy_dir = os.path.join(fields_data, 'fields_npy')
    met = os.path.join(fields_data, 'gridmet')

    mishhape_file = os.path.join(fields_data, 'unexpected_length_fields.json')

    preprocess_historical(pqt_dir, met, gridmet_factors_, npy_dir, target_areas=['108'], overwrite=True,
                          anomalous_recs_file=mishhape_file)

# ========================= EOF ====================================================================
