import os
import json
import pandas as pd


def find_missing_fields(fields_parquet_path, index_dir_path):
    master_fields = set(pd.read_parquet(fields_parquet_path)['OPENET_ID'])
    processed_fields = set()
    index_files = [f for f in os.listdir(index_dir_path) if f.endswith('_index.json')] 
    for fname in index_files:
        with open(os.path.join(index_dir_path, fname), 'r') as f:
            processed_fields.update(json.load(f)['index'])
    missing = master_fields - processed_fields
    print('--- Missing Fields Analysis ---')
    print(f'Initial fields: {len(master_fields)}')
    print(f'Processed fields: {len(processed_fields)}')
    print(f'Missing: {len(missing)}')
    return missing


def investigate_pipeline_losses(fields_parquet_path, summaries_dir_path, anomalous_recs_path, gfid_csv_path):

    mdf = pd.read_parquet(fields_parquet_path)

    print('--- Master File Integrity Checks ---')
    print(f"Fields with null 'Basin' value: {mdf['Basin'].isnull().sum()}")
    print(f"Duplicated OPENET_IDs: {mdf['OPENET_ID'].duplicated().sum()}")

    mdf['basin_code'] = mdf['Basin'].fillna('').apply(lambda s: s.split('_')[0])
    print(f"Fields with unassigned basin_code: {(mdf['basin_code'] == '').sum()}")

    basin_dict = mdf.groupby('basin_code')['OPENET_ID'].apply(list).to_dict()
    if '' in basin_dict:
        del basin_dict['']

    master_basins = set(basin_dict.keys())
    summary_basins = {os.path.basename(s_file).split('_')[0]
                      for s_file in os.listdir(summaries_dir_path) if s_file.endswith('.parquet')}

    basins_without_summaries = master_basins - summary_basins
    if basins_without_summaries:
        print(f"\n--- Basins Missing Summary Files ---")
        print(f"Found {len(basins_without_summaries)} basins in the master file that have no summary file.")
        print(f"  Examples: {list(basins_without_summaries)[:10]}")

    master_fields = set(mdf['OPENET_ID'])
    summary_fields = set()
    total_missing = 0
    print('\n--- Basin-level Summary Mismatches ---')
    for s_file in [f for f in os.listdir(summaries_dir_path) if f.endswith('.parquet')]:
        basin_code = os.path.basename(s_file).split('_')[0]
        df = pd.read_parquet(os.path.join(summaries_dir_path, s_file))
        unq_summary_fields = df['OPENET_ID'].unique()
        unq_master_fields = basin_dict.get(basin_code, [])
        summary_fields.update(unq_summary_fields)
        missing_ct = len(unq_master_fields) - len(unq_summary_fields)
        if missing_ct != 0:
            print(f'Basin {basin_code}: ' 
                  f'Master has {len(unq_master_fields)}, ' 
                  f'Summary has {len(unq_summary_fields)} ' 
                  f'(Missing: {missing_ct})')
            total_missing += abs(missing_ct)

    print(f'\nTotal missing found in summary/master comparison: {total_missing}')

    gfid_map_fields = set(pd.read_csv(gfid_csv_path)['OPENET_ID'])

    with open(anomalous_recs_path, 'r') as f:
        anomalous_fields = set(json.load(f).keys())

    lost_fields = {
        'missing_from_summaries': list(master_fields - summary_fields),
        'missing_gfid_mapping': list(summary_fields - gfid_map_fields),
        'anomalous_record_counts': list(anomalous_fields),
        'basin_to_field_map': basin_dict
    }

    return lost_fields


if __name__ == '__main__':
    root_ = '/media/research/IrrigationGIS'
    if not os.path.exists(root_):
        root_ = '/home/dgketchum/data/IrrigationGIS'

    nv_data_dir_ = os.path.join(root_, 'Nevada', 'dri_field_pts')
    fields_data_ = os.path.join(nv_data_dir_, 'fields_data')
    fields_gis_ = os.path.join(nv_data_dir_, 'fields_gis')
    nv_fields_boundaries_ = os.path.join(fields_gis_, 'Nevada_Agricultural_Field_Boundaries_20250214')

    fields_parquet_ = os.path.join(nv_fields_boundaries_,
                                   'Nevada_Agricultural_Fields_Attrs_20250214_5071_GFID.parquet')
    gfid_csv_ = os.path.join(nv_fields_boundaries_, 'Nevada_Agricultural_Field_Boundaries_20250214_5071_GFID.csv')

    historical_npy_dir_ = os.path.join(fields_data_, 'fields_npy')

    summaries_dir_ = os.path.join(fields_data_, 'parquet_fields')

    anomalous_recs_ = os.path.join(fields_data_, 'unexpected_length_fields.json')

    # find_missing_fields(fields_parquet_, historical_npy_dir_)

    lost_fields_report_ = investigate_pipeline_losses(fields_parquet_,
                                                      summaries_dir_,
                                                      anomalous_recs_,
                                                      gfid_csv_)

    print('\n--- Pipeline Loss Summary ---')
    for category, fields in lost_fields_report_.items():
        if category == 'basin_to_field_map':
            continue
        print(f'Category \'{category}\': Found {len(fields)} fields.')
        if fields and isinstance(fields, list):
            print(f'  Examples: {fields[:5]}')

    basin_map_ = lost_fields_report_.get('basin_to_field_map', {})
    print(f'\n--- Basin Map Summary ---')
    print(f'Found {len(basin_map_)} basins.')
    for basin_code_, field_ids_ in list(basin_map_.items())[:5]:
        print(f'  Basin {basin_code_}: {len(field_ids_)} fields')

# ========================= EOF ====================================================================
