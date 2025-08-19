import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm


def summarize_projections(projection_dir, historical_index_dir, out_dir, metric, target_areas=None):

    os.makedirs(out_dir, exist_ok=True)

    if metric == 'kc':
        hist_col = 'historical_eta_wye'
        future_col_suffix = '_ETa'
    else:
        hist_col = 'historical_netet_wye'
        future_col_suffix = '_netET'

    index_files = [f for f in os.listdir(historical_index_dir) if f.endswith('_index.json')]

    for index_file in tqdm(index_files, desc="Processing Hydrographic Areas"):

        hydro_area = index_file.replace('_index.json', '')

        if target_areas and hydro_area not in target_areas:
            continue

        with open(os.path.join(historical_index_dir, index_file), 'r') as f:
            field_ids = json.load(f)['index']

        mean_historical_water_use = []
        mean_rcp45_projections = []
        mean_rcp85_projections = []

        for field_id in field_ids:
            parquet_file = os.path.join(projection_dir, f'projected_{{metric}}_{field_id}.parquet')
            if not os.path.exists(parquet_file):
                continue

            df = pd.read_parquet(parquet_file)

            if hist_col in df.columns:
                mean_val = df[hist_col].dropna().mean()
                if not pd.isna(mean_val):
                    mean_historical_water_use.append(mean_val)

            rcp45_cols = [c for c in df.columns if 'rcp45' in c and c.endswith(future_col_suffix)]
            if rcp45_cols:
                mean_rcp45 = df[rcp45_cols].dropna(how='all').mean().mean()
                if not pd.isna(mean_rcp45):
                    mean_rcp45_projections.append(mean_rcp45)

            rcp85_cols = [c for c in df.columns if 'rcp85' in c and c.endswith(future_col_suffix)]
            if rcp85_cols:
                mean_rcp85 = df[rcp85_cols].dropna(how='all').mean().mean()
                if not pd.isna(mean_rcp85):
                    mean_rcp85_projections.append(mean_rcp85)

        if not mean_historical_water_use:
            continue

        plt.figure(figsize=(12, 7))

        sns.kdeplot(data=mean_historical_water_use, label='Historical', fill=True, alpha=0.5)

        if mean_rcp45_projections:
            sns.kdeplot(data=mean_rcp45_projections, label='RCP 4.5 Projection', linewidth=2)

        if mean_rcp85_projections:
            sns.kdeplot(data=mean_rcp85_projections, label='RCP 8.5 Projection', linewidth=2)

        plt.title(f'Distribution of Mean Water Use ({{metric.upper()}}) Hydro Area: {hydro_area}')
        plt.xlabel(f'Mean Annual Water Use ({{future_col_suffix.replace("_", "")}})')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)

        hist_path = os.path.join(out_dir, f'{hydro_area}_{metric}_hist.png')
        plt.savefig(hist_path)
        plt.close()

        summary_df = pd.DataFrame({
            'historical': pd.Series(mean_historical_water_use),
            'rcp45': pd.Series(mean_rcp45_projections),
            'rcp85': pd.Series(mean_rcp85_projections)
        })
        summary_stats = summary_df.describe()
        stats_path = os.path.join(out_dir, f'{hydro_area}_{{metric}}_stats.csv')
        summary_stats.to_csv(stats_path)
        print(f'Wrote {hist_path} and {stats_path}')


def plot_aggregated_projections(projection_dir, historical_index_dir, out_dir, metric,
                                target_areas=None, fields_table=None, fips_filter=None, aggregate_all=False):

    os.makedirs(out_dir, exist_ok=True)

    if metric == 'kc':
        hist_col = 'historical_eta_wye'
        future_col_suffix = '_ETa'
    else:
        hist_col = 'historical_netet_wye'
        future_col_suffix = '_netET'

    field_ids_from_table = None
    if fields_table:
        fields = pd.read_parquet(fields_table)
        if fips_filter:
            fields = fields[fields['FIPS'] == fips_filter]
        field_ids_from_table = set(fields['OPENET_ID'].tolist())

    all_hydro_areas = {}
    index_files = [f for f in os.listdir(historical_index_dir) if f.endswith('_index.json')]
    for index_file in index_files:
        hydro_area = index_file.replace('_index.json', '')
        if target_areas and hydro_area not in target_areas:
            continue
        with open(os.path.join(historical_index_dir, index_file), 'r') as f:
            all_hydro_areas[hydro_area] = json.load(f)['index']

    def _aggregate_and_plot(field_ids, plot_title, out_file):
        if not field_ids:
            print(f"No fields to process for {plot_title}")
            return

        total_historical_series = None
        total_projections = {}

        for field_id in field_ids:
            parquet_file = os.path.join(projection_dir, f'projected_{metric}_{field_id}.parquet')
            if not os.path.exists(parquet_file):
                continue

            df = pd.read_parquet(parquet_file)

            if hist_col in df.columns:
                historical_series = df[hist_col].dropna()
                if total_historical_series is None:
                    total_historical_series = historical_series
                else:
                    total_historical_series = total_historical_series.add(historical_series, fill_value=0)

            for col in df.columns:
                if future_col_suffix in col and ('rcp45' in col or 'rcp85' in col):
                    projection_series = df[col].dropna()
                    if col not in total_projections:
                        total_projections[col] = projection_series
                    else:
                        total_projections[col] = total_projections[col].add(projection_series, fill_value=0)

        if total_historical_series is None and not total_projections:
            return

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(15, 8))

        if total_historical_series is not None:
            ax.plot(total_historical_series.index, total_historical_series, label='Total Historical', color='black',
                    linewidth=2.5)

        colors = {'rcp45': 'dodgerblue', 'rcp85': 'firebrick'}
        for scenario in ['rcp45', 'rcp85']:
            scenario_cols = [c for c in total_projections.keys() if scenario in c]
            if not scenario_cols:
                continue

            scenario_df = pd.DataFrame(total_projections)[scenario_cols]
            mean_projection = scenario_df.mean(axis=1)
            std_projection = scenario_df.std(axis=1)
            ax.plot(mean_projection.index, mean_projection,
                    label=f'Mean {scenario.upper()}', color=colors[scenario], linestyle='--')
            ax.fill_between(mean_projection.index, mean_projection - std_projection, mean_projection + std_projection,
                            color=colors[scenario], alpha=0.2, label=f'{scenario.upper()} Range (±1σ)')

        ax.set_title(plot_title, fontsize=16)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel(f'Total Annual Water Use ({future_col_suffix.replace("_", "")})', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(out_file, dpi=300)
        plt.close(fig)
        print(f'Wrote aggregated plot to {out_file}')

    if aggregate_all:
        master_field_list = []
        for fields in all_hydro_areas.values():
            master_field_list.extend(fields)

        if field_ids_from_table is not None:
            final_field_ids = list(set(master_field_list) & field_ids_from_table)
        else:
            final_field_ids = master_field_list

        title = 'Aggregated Historical and Projected Water Use'
        if fips_filter:
            title += f'\nFIPS: {fips_filter}; {len(master_field_list)} fields'
        elif target_areas:
            title += f'\nBasins: {", ".join(target_areas)}; {len(master_field_list)} fields'

        out_fname = f'aggregated_all_{metric}_projections.png'
        if fips_filter:
            out_fname = f'aggregated_fips_{fips_filter}_{metric}_projections.png'

        _aggregate_and_plot(final_field_ids, title, os.path.join(out_dir, out_fname))
    else:
        for hydro_area, field_ids in all_hydro_areas.items():
            if field_ids_from_table is not None:
                final_field_ids = list(set(field_ids) & field_ids_from_table)
            else:
                final_field_ids = field_ids

            title = f'Aggregated Water Use\nBasin: {hydro_area}'
            if fips_filter:
                title += f', FIPS: {fips_filter}'

            out_fname = f'aggregated_{hydro_area}_{metric}_projections.png'
            _aggregate_and_plot(final_field_ids, title, os.path.join(out_dir, out_fname))


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    nv_data_dir = os.path.join(root, 'Nevada', 'dri_field_pts')
    fields_gis = os.path.join(nv_data_dir, 'fields_gis')
    nv_fields_boundaries = os.path.join(fields_gis, 'Nevada_Agricultural_Field_Boundaries_20250214')

    historical_npy_dir_ = os.path.join(nv_data_dir, 'fields_data', 'fields_npy')
    fields_shp_ = os.path.join(nv_fields_boundaries, 'Nevada_Agricultural_Fields_Attrs_20250214_5071_GFID.parquet')

    calculation_type = 'kc'

    projection_out_dir_ = os.path.join(nv_data_dir, 'fields_data', 'iwu_projections', calculation_type)
    postprocess_out_dir_ = os.path.join(nv_data_dir, 'fields_data', 'postprocess', 'aggregated_plots')

    target_basins = ['196', '180', '183', '181', '201', '172', '208', '202', '200', '170', '199', '198', '209',
                     '210', '203', '204', '205', '222']

    # summarize_projections(projection_dir=projection_out_dir_,
    #                       historical_index_dir=historical_npy_dir_,
    #                       out_dir=postprocess_out_dir_,
    #                       metric=calculation_type,
    #                       target_areas=target_basins)

    plot_aggregated_projections(projection_dir=projection_out_dir_,
                                historical_index_dir=historical_npy_dir_,
                                out_dir=postprocess_out_dir_,
                                metric=calculation_type,
                                target_areas=target_basins,
                                fields_table=fields_shp_,
                                fips_filter='32017',
                                aggregate_all=True)

# ========================= EOF ====================================================================
