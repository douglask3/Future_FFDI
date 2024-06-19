import os
import sys
os.environ['USE_PYGEOS'] = '0'

import dask.array.ma
import iris
import glob
import pandas as pd
import numpy as np
import numpy.ma as ma
import cftime
from iris.util import unify_time_units
import geopandas as gpd
sys.path.append('/net/home/h02/hadhy/PycharmProjects/impactstoolbox')
from impactstoolbox.tools import helper_functions as hf
sys.path.append('/home/h02/hadhy/GitHub/wcssp_casestudies')
import std_functions as sf
import subprocess
import warnings
warnings.simplefilter('ignore')
import pdb


def get_ens_df():
    #ensemble members for rcp2.6 historical runs
    hist_rcp2_6=['akowa', 'akowb', 'akowc', 'akowe', 'akowf', 'akowh', 'akowi', 'akowj', 'akowk', 'akowl',
                'akowm', 'akown', 'akowo', 'akowp', 'akowq', 'akoxa', 'akoxb', 'akoxc', 'akoxd', 'akoxe',
                'akoxg', 'akoxi', 'akoxj', 'akoxl', 'akoxm', 'akoxn', 'akoxo', 'akoxp', 'akoxq', 'akoya',
                'akoyb', 'akoyc', 'akoye', 'akoyf', 'akoyg', 'akoyh', 'akoyi', 'akoyj', 'akoyk', 'akoyl',
                'akoym', 'akoyn', 'akoyq', 'akoza', 'akozb', 'akozc', 'akozd', 'akoze', 'akozg', 'akozh',
                'akozi', 'akozj', 'akozl', 'akozn', 'akozo', 'akozp', 'akozq']
    #ensemble members for rcp2.6 future runs
    fut_rcp2_6=['akuka', 'akukb', 'akukc', 'akuke', 'akukf', 'akukh','akuki', 'akukj', 'akukk', 'akukl',
                'akukm', 'akukn', 'akuko', 'akukp', 'akukq', 'akula', 'akulb', 'akulc', 'akuld', 'akule',
                'akulg', 'akuli', 'akulj', 'akull', 'akulm', 'akuln', 'akulo', 'akulp', 'akulq', 'akuma',
                'akumb', 'akumc', 'akume', 'akumf', 'akumg', 'akumh', 'akumi', 'akumj', 'akumk', 'akuml',
                'akumm', 'akumn', 'akumq', 'akuna', 'akunb', 'akunc', 'akund', 'akune', 'akung', 'akunh',
                'akuni', 'akunj', 'akunl', 'akunn', 'akuno', 'akunp', 'akunq']
    #ensemble members for rcp8.5 historical runs (same as for rcp2.6)
    hist_rcp8_5=['akowa', 'akowb', 'akowc', 'akowe', 'akowf', 'akowh', 'akowi', 'akowj', 'akowk', 'akowl',
                 'akowm', 'akown', 'akowo', 'akowp', 'akowq', 'akoxa', 'akoxb', 'akoxc', 'akoxd', 'akoxe',
                 'akoxg', 'akoxi', 'akoxj', 'akoxl', 'akoxm', 'akoxn', 'akoxo', 'akoxp', 'akoxq', 'akoya',
                 'akoyb', 'akoyc', 'akoye', 'akoyf', 'akoyg', 'akoyh', 'akoyi', 'akoyj', 'akoyk', 'akoyl',
                 'akoym', 'akoyn', 'akoyq', 'akoza', 'akozb', 'akozc', 'akozd', 'akoze', 'akozg', 'akozh',
                 'akozi', 'akozj', 'akozl', 'akozn', 'akozo', 'akozp', 'akozq']
    #ensemble members for rcp8.5 future runs
    fut_rcp8_5=['aldpa', 'aldpb', 'aldpc', 'aldpe', 'aldpf', 'aldph', 'aldpi', 'aldpj', 'aldpk', 'aldpl',
                'aldpm', 'aldpn', 'aldpo', 'aldpp', 'aldpq', 'aldqa', 'aldqb', 'aldqc', 'aldqd', 'aldqe',
                'aldqg', 'aldqi', 'aldqj', 'aldql', 'aldqm', 'aldqn', 'aldqo', 'aldqp', 'aldqq', 'aldra',
                'aldrb', 'aldrc', 'aldre', 'aldrf', 'aldrg', 'aldrh', 'aldri', 'aldrj', 'aldrk', 'aldrl',
                'aldrm', 'aldrn', 'aldrq', 'aldsa', 'aldsb', 'aldsc', 'aldsd', 'aldse', 'aldsg', 'aldsh',
                'aldsi', 'aldsj', 'aldsl', 'aldsn', 'aldso', 'aldsp', 'aldsq']

    # Create a pandas dataframe from the above lists
    df2p6 = pd.DataFrame({'rcp': 'rcp2.6', 'hist': hist_rcp2_6, 'fut': fut_rcp2_6})
    df8p5 = pd.DataFrame({'rcp': 'rcp8.5', 'hist': hist_rcp8_5, 'fut': fut_rcp8_5})
    df = pd.concat([df2p6, df8p5], ignore_index=True)
    df['gwl_1_5_start'] = np.nan
    df['gwl_2_start'] = np.nan
    df['gwl_4_start'] = np.nan
    df = df.set_index('fut')

    # Now read the GWL data
    gwl_dir = '/data/users/hadhy/ESMS/fire_paper/GlobalWarmingLevels/*/*.txt'
    gwl_files = glob.glob(gwl_dir)
    for f in gwl_files:
        rcp = os.path.dirname(f).split('/')[-1].replace('_', '.')
        gwl = 'gwl_' + os.path.basename(f).split('gwl_')[1].split('.txt')[0].replace('_deg', '')
        print(rcp, gwl)
        # Read the csv file
        gwldf = pd.read_csv(f, header=None, names=['fut', gwl+'_start', 'end_yr'], na_values=['nan', ' nan', ' nan '], dtype={'fut': 'string', gwl+'_start':'Int64', 'end_yr':'Int64'})
        gwldf = gwldf.set_index('fut')
        # Update df with the contents of gwldf
        df.update(gwldf)

    # Removes the index 'fut', so it is part of the table again
    df.reset_index(inplace=True)
    # Reorder the columns to something more sensible
    df = df.reindex(columns=['rcp', 'hist', 'fut', 'gwl_1_5_start', 'gwl_2_start', 'gwl_4_start'])

    return df


def check_ens_numbers():
    'do stuff'
    indir = '/scratch/hadin/fire/ffdi_inputs/'
    files = sorted(glob.glob(indir+'*/*'))
    ens = sorted(list(set([os.path.basename(f).split('_')[5] for f in files])))
    dbl = 0
    rcp2p6 = 0
    rcp8p5 = 0
    for em in ens:
        emfiles = glob.glob(indir+'*/*'+em+'*')
        filercps = set([os.path.dirname(emf).split('/')[-1] for emf in emfiles])
        print(em, ':', filercps)
        dbl = dbl + 1 if len(filercps) > 1 else dbl
        rcp2p6 = rcp2p6 + 1 if filercps == {'rcp2_6'} else rcp2p6
        rcp8p5 = rcp8p5 + 1 if filercps == {'rcp8_5'} else rcp8p5

    print('dbl:', dbl, '; rcp2.6:', rcp2p6, '; rcp8.5:', rcp8p5)


def load_files(file_list, start_yr):

    if np.isnan(start_yr):
        return None
    else:
        varnames = ['relative_humidity', 'air_temperature', 'precipitation_in_mpersec', 'windspeed_in_km_per_hr', 'soil_moisture_content_top_metre'] # 'vol_SMC_at_field_capacity'
        start_dt = cftime.Datetime360Day(start_yr, 1, 1)
        end_dt = cftime.Datetime360Day(start_yr + 20, 1, 1)
        cubelist = iris.cube.CubeList([])
        for var in varnames:
            print('   ... ' + var, end=' ... ')
            cubes = iris.load(file_list, iris.Constraint(name=var))
            unify_time_units(cubes)
            varcubes = iris.cube.CubeList([])
            for cube in cubes:
                cube.remove_coord('forecast_reference_time')
                varcubes.append(cube)

            varcube = varcubes.concatenate_cube()
            varcube = sf.periodConstraint(varcube, start_dt, end_dt)

            cubelist.append(varcube)
            print('done')

        return cubelist


def get_variables_by_ffdi_theshold(df, mint, maxt, ofile):
    """
    Gets a massive dataframe of FFDI value (>=12) with each variable per grid cell per day
    :param df: Dataframe that tells us for each ens member what the start year is for each GWL
    :param mint: masks < mint
    :param maxt: if no maximum, use np.inf
    :return:
    """

    # Root dir of the data
    indir = '/scratch/hadhy/fire/ffdi_input_variables/'

    # Baseline start year
    bl_start_yr = 1986

    vars = ['relative_humidity', 'soil_moisture_content_top_metre', 'precipitation_in_mpersec',
            'windspeed_in_km_per_hr', 'air_temperature']

    # Make empty output dataframe
    column_names = ['t', 'x', 'y', 'gwl', 'ensemble_member', 'rcp', 'FFDI', 'air_temperature', 'precipitation_in_mpersec', 'relative_humidity', 'soil_moisture_content_top_metre', 'windspeed_in_km_per_hr']
    df_final = pd.DataFrame(columns=column_names)

    # Loop through each row of the table (contains the RCP & future jobid)
    for i, row in df.iterrows():
        print(i, row['rcp'], row['hist'], row['fut'])
        rcp = row['rcp'].replace('.', '_')

        # ofile = f'/data/users/hadhy/ESMS/fire_paper/data/odf_bigdata_country_{mint}_to_{maxt}_temp{i}.csv'
        ofile_temp = ofile.replace('.csv', f'_temp{i}.csv')
        if os.path.isfile(ofile_temp):
            df_final = pd.read_csv(ofile_temp)
            continue
        else:
            last_ofile = ofile.replace('.csv', f'_temp{i-1}.csv')
            # last_ofile = f'/data/users/hadhy/ESMS/fire_paper/data/odf_bigdata_country_{mint}_to_{maxt}_temp{i-1}.csv'
            if os.path.isfile(last_ofile):
                df_final = pd.read_csv(last_ofile)

            ##########
            # Get the pre-calculated FFDI (saved as a timeseries of baseline + future, based on future ens_member_id
            print('   Loading FFDI ... ')
            ffdi_filelist = glob.glob('/scratch/hadhy/fire/ffdi_output/'+rcp+'/ffdi_'+rcp+'_'+row['hist']+'*.nc')
            ffdi_filelist.extend(glob.glob('/scratch/hadhy/fire/ffdi_output/'+rcp+'/ffdi_'+rcp+'_'+row['fut']+'*.nc'))

            # Load the files into an iris cubelist
            ffdi_cubes = iris.load(ffdi_filelist)  # includes baseline and future FFDI

            # Remove some things to allow for the cube to be concatenated
            for j in range(len(ffdi_cubes)):
                ffdi_cubes[j].remove_coord('forecast_period')
                ffdi_cubes[j].remove_coord('year')

            # Now unify time coords and concatenate
            unify_time_units(ffdi_cubes)
            ffdi_cube = ffdi_cubes.concatenate_cube()

            # Get the burnable mask
            burnable = hf.get_burnable_mask(ffdi_cube)

            # Now get FFDI variables
            var_filelist = sorted(glob.glob(f"{indir}{rcp}/ffdi_input_variables_{rcp}_{row['hist']}_*"))
            var_filelist.extend(sorted(glob.glob(f"{indir}{rcp}/ffdi_input_variables_{rcp}_{row['fut']}_*")))

            # Make a dictionary of cubes for all variables  ...
            all_var_cubes = {}
            for v in vars:
                print(f"   ... {v}")

                # Do all the same loading things ...
                var_cubes = iris.load(var_filelist, v)  # baseline and future variable
                for j in range(len(var_cubes)):
                    for coord in ['forecast_reference_time', 'forecast_period', 'year']:
                        try:
                            var_cubes[j].remove_coord(coord)
                        except:
                            continue

                unify_time_units(var_cubes)
                all_var_cubes[v] = var_cubes.concatenate_cube()

            # Time subset of ffdi and variables
            tcon_dict = {}
            # Baseline
            hist_start = cftime.Datetime360Day(bl_start_yr, 1, 1)
            hist_end = cftime.Datetime360Day(bl_start_yr + 20, 1, 1)
            hist_tcon = iris.Constraint(time=lambda cell: hist_start <= cell.point < hist_end)
            tcon_dict['Baseline'] = hist_tcon

            # GWL 1.5C
            gwl1p5_start = cftime.Datetime360Day(row['gwl_1_5_start'], 1, 1)
            gwl1p5_end = cftime.Datetime360Day(row['gwl_1_5_start'] + 20, 1, 1)
            gwl1p5_tcon = iris.Constraint(time=lambda cell: gwl1p5_start <= cell.point < gwl1p5_end)
            tcon_dict['GWL1.5'] = gwl1p5_tcon

            # GWL 2C
            try:
                gwl2_start = cftime.Datetime360Day(row['gwl_2_start'], 1, 1)
                gwl2_end = cftime.Datetime360Day(row['gwl_2_start'] + 20, 1, 1)
                gwl2_tcon = iris.Constraint(time=lambda cell: gwl2_start <= cell.point < gwl2_end)
                tcon_dict['GWL2'] = gwl2_tcon
            except:
                print(f"{row['rcp']} & {row['fut']} doesn\'t reach 2C GWL")
                pass

            # GWL 4C
            try:
                gwl4_start = cftime.Datetime360Day(row['gwl_4_start'], 1, 1)
                gwl4_end = cftime.Datetime360Day(row['gwl_4_start'] + 20, 1, 1)
                gwl4_tcon = iris.Constraint(time=lambda cell: gwl4_start <= cell.point < gwl4_end)
                tcon_dict['GWL4'] = gwl4_tcon
            except:
                print(f"{row['rcp']} & {row['fut']} doesn\'t reach 4C GWL")
                pass

            # Loop through the time constraints and extract all the grid cells with data
            print(f'   Extracting time slices')
            dfout = {}
            for key, tcon in tcon_dict.items():
                print(f'   ... {key}: FFDI, ')
                # Extract the days in the time constraint
                ffdi_tconned = ffdi_cube.extract(tcon)
                # Mask according to the min and max thresholds
                ffdi_tconned.data = ma.masked_where((ffdi_tconned.data < mint) | (ffdi_tconned.data >= maxt), ffdi_tconned.data)
                # Combined burnable mask with masked data
                burnable_allts = np.broadcast_to(ma.getmask(burnable.data), ffdi_tconned.shape)
                fullmask = burnable_allts | ma.getmask(ffdi_tconned.data)
                # Do the masking and convert to pandas dataframe
                tmp = mask2df(ffdi_tconned, fullmask, 'FFDI', key, row['fut'], row['rcp'])

                if isinstance(dfout, dict):
                    dfout = tmp.copy()
                else:
                    dfout = pd.concat([dfout, tmp])

                for k in all_var_cubes.keys():
                    print(f"   ... {key}: {k}")
                    var_tconned = all_var_cubes[k].extract(tcon)
                    varout = mask2df(var_tconned, fullmask, k, key, row['fut'], row['rcp'])
                    # varout = mask2df(var_tconned, ~ma.getmask(ffdi_tconned.data), k, key, row['fut'], row['rcp'])
                    dfout = pd.concat([dfout, varout])

            # NB: dfout contains all the GWLs (i.e. time slices) for this ens member
            # This pivot widens the var column, to give a column per category
            # Therefore each row is a unique combination of t-x-y-gwl
            odf_wide = pd.pivot_table(dfout, values="value", index=['t', 'x', 'y', 'gwl', 'names', 'ensemble_member', 'rcp'], columns=["var"])
            odf_wide.reset_index(inplace=True)
            ### Probably need to sample here, in order to reduce the size of the output ###
            # odf_wide_ss = odf_wide.sample(n=10000, replace=False, axis='index', ignore_index=True)
            odf_wide_ss = odf_wide.groupby(['gwl', 'names']).sample(frac=0.001, replace=False)  # axis='index', ignore_index=True
            # odf_wide_ss = odf_wide.groupby(['gwl', 'names']).sample(n=100, replace=True)
            # Concatenate with existing output, and on to the next ensemble member
            df_final = pd.concat([df_final, odf_wide_ss])
            # Now save the current state of df_final to a temporary file
            df_final.to_csv(ofile_temp)

    df_final.to_csv(ofile)

    return df_final


def mask2df(cube, mask_arr, varname, gwlname, ensmem, rcp, region_name='natural_earth_v5_0_0.countries_110'):
    """
    Mask the cube data using the mask, and creates a pandas dataframe using the additional metadata
    :param cube: 3D cube containing the data to be extracted and masked
    :param mask_arr: 3D numpy.ma mask (assumes mask=False for cells we want to keep)
    :param varname: variable name
    :param gwlname: global warming level name
    :param ensmem: ensemble members name
    :param rcp: RCP name
    :param region_name: could be 'ipcc' or anything accepted by regionmask. Options include all those available in regionmask: ['ar6.all', 'ar6.land', 'ar6.ocean', 'natural_earth_v5_0_0.countries_110', 'natural_earth_v5_0_0.countries_50', 'natural_earth_v5_0_0.countries_10','natural_earth_v5_0_0.land_110', 'natural_earth_v5_0_0.ocean_basins_50']
    :return: pandas dataframe
    """
    import regionmask

    cells_we_want = ~mask_arr
    value = ma.getdata(cube.data)[cells_we_want].flatten()

    myu = cube.coord('time').units
    t = cube.coord('time').points
    y = cube.coord('latitude').points
    x = cube.coord('longitude').points
    ts, ys, xs = np.meshgrid(t, y, x, sparse=False, indexing='ij')

    ts_out = ts[cells_we_want].flatten()
    xs_out = [((x + 180) % 360) - 180 for x in xs[cells_we_want].flatten()]  # Corrects for 0 to 360 problem
    ys_out = ys[cells_we_want].flatten()
    points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x=xs_out, y=ys_out), crs="EPSG:4326")

    if region_name == 'ipcc':
        inshp = '/net/home/h02/hadhy/PycharmProjects/impactstoolbox/impactstoolbox/data/IPCC-WGI-reference-regions-v4.shp'
        shp = gpd.read_file(inshp, crs="epsg:4326")
    else:
        # region_name = 'natural_earth_v5_0_0.countries_110'
        regions = getattr(getattr(regionmask.defined_regions, region_name.split('.')[0]), region_name.split('.')[1])
        shp = regions.to_geodataframe()
        shp = shp.set_crs(epsg=4326, allow_override=True)
        shp = shp[shp['names'].isin(['Australia', 'United States of America', 'Brazil'])]

    sj = points.sjoin(shp, how="left")
    sj.reset_index(inplace=True)
    sj.drop_duplicates(subset=['index', 'geometry'], keep='first', inplace=True)

    # The most likely cause of problems could be due to the joined data not having the same number of records as
    # other records
    # print(len(xs_out))
    # print(len(sj['Acronym'].to_list()))
    if region_name == 'ipcc':
        dfout = pd.DataFrame({'t': myu.num2date(ts_out), 'x': xs_out, 'y': ys_out, 'value': value, 'var': varname,
                              'names': sj['Acronym'].to_list(), 'gwl': gwlname, 'ensemble_member': ensmem, 'rcp': rcp})  #
    else:
        dfout = pd.DataFrame({'t': myu.num2date(ts_out), 'x': xs_out, 'y': ys_out, 'value': value, 'var': varname,
                              'names': sj['names'].to_list(), 'gwl': gwlname, 'ensemble_member': ensmem,
                              'rcp': rcp})  #

    return dfout


def get_data_to_plot(type, bl_start_yr, df, climatology=True):
    '''

    :param type: Could be any one of Baseline, GWL1.5, GWL2, GWL4, GWL1.5_change, GWL2_change, GWL4_change
    :param bl_start_yr: baseline start year usually 1986
    :param df: Simple dataframe of the start year for each GWL and hist & fut jobid
    :param climatology: boolean to say whether to create a climatology for each grid cell,
    or to just export to csv the full daily data (useful for correlation plots)
    :return:
    '''

    # Root dir of the data
    indir = '/scratch/hadhy/fire/ffdi_input_variables/'
    # indir = '/scratch/hadin/fire/ffdi_inputs/'

    # Create an empty variables
    country_cube = None
    hist_cubes = []
    fut_cubes = []

    # GWL LUT
    gwl_lut = {'GWL1.5': 'gwl_1_5_start',
               'GWL2': 'gwl_2_start',
               'GWL4': 'gwl_4_start'}

    # Make empty output dataframe
    column_names = ['Date', 'Ensemble_Member', 'Country', 'GWL', 'FFDI', 'RH', 'Temperature', 'Precipitation', 'Wind_Speed', 'Soil_Moisture']
    odf = pd.DataFrame(columns=column_names)

    # Loop through each row of the table (contains the RCP & future jobid)
    for i, row in df.iterrows():
        print(i, row['rcp'], row['hist'], row['fut'])
        rcp = row['rcp'].replace('.', '_')

        ##########
        # Get the pre-calculated FFDI (saved as a timeseries of baseline + future, based on future ens_member_id
        print('   Loading FFDI ... ', end='')
        ffdi_file = '/scratch/hadhy/fire/ffdi_output_joined/'+rcp+'/ffdi_joined_'+rcp+'_'+row['fut']+'.nc'
        print(ffdi_file)
        ffdi_cube = iris.load_cube(ffdi_file) # baseline and futurev
        # Time subset of ffdi
        # NB: Even for future plots, we still need the baseline FFDI >= 24 mask
        hist_start = cftime.Datetime360Day(bl_start_yr, 1, 1)
        hist_end = cftime.Datetime360Day(bl_start_yr + 20, 1, 1)
        hist_ffdi = sf.periodConstraint(ffdi_cube, hist_start, hist_end)
        ffdi_vh_count = hist_ffdi.collapsed('time', iris.analysis.COUNT, function=lambda values: values >= 24)
        print('done')

        ##########
        # Get a gridded map of the country mask
        if country_cube is None:
            print('   Loading Country Mask ... ', end='')
            country_cube = get_country_shapefile(ffdi_vh_count)
            print('done')

        ##########
        # Get the data requested
        if type == 'Baseline' or 'change' in type:
            print('   Loading historical cubes ... ')
            hist_files = sorted(glob.glob(indir + rcp + '/*' + row['hist'] + '*'))
            hist_cubes = load_files(hist_files, bl_start_yr)
            print('   done')

        if 'GWL' in type:
            print('   Loading future cubes ... ')
            # Get the column name for this GWL, and check that it is not NA
            col = gwl_lut[type.split('_')[0]]
            if np.isnan(row[col]):
                # Means that this ens member didn't reach this GWL, so move to the next ens member
                continue
            fut_files = sorted(glob.glob(indir + rcp + '/*' + row['fut'] + '*'))
            fut_cubes = load_files(fut_files, row[col])
            # Time subset of ffdi
            fut_start = cftime.Datetime360Day(row[col], 1, 1)
            fut_end = cftime.Datetime360Day(row[col] + 20, 1, 1)
            fut_ffdi = sf.periodConstraint(ffdi_cube, fut_start, fut_end)
            print('   done')

        ##########
        # Next, do the masking of FFDI, Countries and extract into a dataframe
        if hist_cubes and fut_cubes:
            # Ignore for the moment
            # need to calculate change in some way
            # Maybe per pixel climatologies of seasons DJF, MAM, JJA, SON?
            # Maybe annual means per pixel
            print('Nothing to do here yet')

        elif hist_cubes and not fut_cubes:
            print('   Masking & extracting historical cubes ... ')
            # Baseline only
            odf = mask_and_extract(hist_cubes, hist_ffdi, ffdi_vh_count, country_cube, row['fut'], type, row['rcp'], odf)
            print('   done')

        elif not hist_cubes and fut_cubes:
            print('   Masking & extracting future cubes ... ')
            # Future absolute
            odf = mask_and_extract(fut_cubes, fut_ffdi, ffdi_vh_count, country_cube, row['fut'], type, row['rcp'], odf)
            print('   done')

        else:
            # nothing in hist_cubes or fut_cubes, so there's a problem
            return None

        # Write out temporary odf file for testing plotting
        try:
            odf.drop(columns='Unnamed: 0', inplace=True)
        except:
            print('Don\'t worry about it')

        # pdb.set_trace()

        if climatology:
            # TODO: Check what this line does, and does it introduce any inconsistencies esp for GWL1.5
            cell_climatologies = odf.groupby(['CellX', 'CellY', 'Ensemble_Member', 'Country', 'GWL', 'RCP']).agg(
                {'FFDI': 'mean', 'RH': 'mean', 'Temperature': 'mean', 'Precipitation': 'mean', 'Wind_Speed': 'mean',
                 'Soil_Moisture': 'mean'})
            cell_climatologies.reset_index(inplace=True)
            cell_climatologies.to_csv('/data/users/hadhy/ESMS/fire_paper/odf_climatology_'+str(i)+'_'+type+'.csv', index=False)
        else:
            odf.to_csv('/data/users/hadhy/ESMS/fire_paper/odf_allcells_'+str(i)+'_'+type+'.csv', index=False)

    # return cell_climatologies


def mask_and_extract(cubelist, ffdi, ffdi_vh_count, country_cube, ens, type, rcp, odf, ffdi_threshold=20):
    '''
    Uses the ffdi_mask and country_cube to mask data from cubelist and
        puts it into a dataframe that can be used for plotting
    :param cubelist: Each cube contains the variables needed to calculate the FFDI
    :param ffdi: Daily FFDI during the period
    :param ffdi_vh_count: Count of grid cells with >1 day per year Very High FFDI in the baseline period, to use as a mask
    :param country_cube: Grid of countries (Australia, Brazil and USA) for masking
    :param ens: ensemble member
    :param rcp: rcp
    :param type: baseline, GWL1.5, GWL2, GWL4, GWL1.5_change, GWL2_change, or GWL4_change
    :param odf: Big dataframe for plotting. Columns = ['Ensemble_Member', 'Country', 'GWL', 'FFDI', 'RH', 'Temperature',
       'Precipitation', 'Wind_Speed', 'Soil_Moisture']
    :return: pandas dataframe for plotting
    '''


    col_lut = {'relative_humidity': 'RH', 'air_temperature': 'Temperature', 'precipitation_in_mpersec': 'Precipitation', 'windspeed_in_km_per_hr': 'Wind_Speed', 'soil_moisture_content_top_metre': 'Soil_Moisture'}

    for k, v in country_cube.attributes.items():

        # Mask using FFDI timeseries
        print('   ... Applying FFDI and Country mask ... ' + k)

        columns = ['Ensemble_Member', 'Country', 'GWL', 'variable', 'value', 'Date', 'RCP', 'CellX', 'CellY']
        results = pd.DataFrame(columns=columns)

        ffdi_cntry = ffdi.copy()
        # Get 2D mask of cells with < 20 V. High FFDI days from baseline period for this country
        data2d = ma.masked_where((country_cube.data.data != v) | (ffdi.data.data < ffdi_threshold), ffdi.data.data)
        # Broadcast the 2D mask to the ffdi (daily) cube shape
        basemask = np.broadcast_to(data2d.mask, ffdi.shape)
        # Add to the mask the days in the timeseries when FFDI < 24
        newmask = ffdi_cntry.data.mask | basemask
        # Adds the mask to the cube
        ffdi_cntry.data.mask = newmask
        # Gets the values that are not masked
        vals = ma.compressed(ffdi_cntry.data)
        # Gets the time coordinates for each grid cell that is not masked
        myu = ffdi_cntry.coord('time').units
        dates = myu.num2date(ffdi_cntry.coord('time').points[np.where(ffdi_cntry.data.mask == False)[0]])
        # Gets the cell coordinates where we have data
        cellx, celly = np.where(ffdi_cntry.data.mask == False)[1:]
        # Puts the FFDI values in the results dataframe
        these_results = pd.DataFrame({columns[0]: ens,
                                      columns[1]: k,
                                      columns[2]: type,
                                      columns[3]: 'FFDI',
                                      columns[4]: vals,
                                      columns[5]: dates,
                                      columns[6]: rcp,
                                      columns[7]: cellx,
                                      columns[8]: celly})
        results = pd.concat([results, these_results])
        
        for cube in cubelist:

            cube.data.mask = newmask
            vals = ma.compressed(cube.data)
            myu = cube.coord('time').units
            dates = myu.num2date(cube.coord('time').points[np.where(cube.data.mask == False)[0]])
            cellx, celly = np.where(cube.data.mask == False)[1:]
            these_results = pd.DataFrame({columns[0]: ens,
                                          columns[1]: k,
                                          columns[2]: type,
                                          columns[3]: col_lut[cube.name()],
                                          columns[4]: vals,
                                          columns[5]: dates,
                                          columns[6]: rcp,
                                          columns[7]: cellx,
                                          columns[8]: celly})
            results = pd.concat([results, these_results])

        # Change the pandas DataFrame to wide format (i.e. variables are now columns)
        try:
            df_pivot = results.pivot_table(index=[columns[0], columns[1], columns[2], columns[5], columns[6], columns[7], columns[8]], columns='variable', values='value').reset_index()
            odf = pd.concat([odf, df_pivot])
        except:
            print('Unable to add results to odf')

    return odf


def get_country_shapefile(cube):
    '''
    Get Country Masks (NB: has to switch to -180 to 180, and then back to 0 to 360)
    USA=4, BRA=5, AUS=1
    :param cube: iris.cube.Cube - has to be 2D (longitude, latitude coordinates only)
    :return: cube with the countries burnt into it
    '''

    fire_paper_countries = '/data/users/hadhy/Projects/LandCoverCCI/Data/fire_paper_countries.shp'

    country_cube = sf.poly2cube(fire_paper_countries, 'MAPCOLOR7', cube.intersection(longitude=(-180, 180)))
    country_cube = country_cube.intersection(longitude=(0, 360))
    country_cube.coord('longitude').circular = True
    country_cube.data = ma.masked_equal(country_cube.data, 0)
    country_cube.attributes = {'Australia': 1, 'USA': 4, 'Brazil': 5}

    return country_cube


def plot_gwl_year(ensdf):

    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style="whitegrid")
    sns.set_context("paper")
    # sns.set_palette('bright')
    sns.set_palette(sns.color_palette('bright')[1:3])

    ensdf.columns = ['rcp', 'hist', 'fut', '1.5 Degrees', '2 Degrees', '4 Degrees']
    ensdflong = pd.melt(ensdf, id_vars=['rcp', 'hist', 'fut'], value_vars=['1.5 Degrees', '2 Degrees', '4 Degrees'])
    ensdflong['Median Year'] = ensdflong['value'] + 10
    ax = sns.boxplot(data=ensdflong, y='variable', x='Median Year', hue='rcp', whis=[0,100])
    ax = sns.stripplot(data=ensdflong, y='variable', x='Median Year', hue='rcp', palette={'rcp2.6': 'Black', 'rcp8.5': 'black'}, dodge=True, size=3)
    ax.set(xlim=(2000, 2100))

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[0:2], labels[0:2])

    plt.savefig('/data/users/hadhy/ESMS/fire_paper/plots/gwls_by_rcp.png')


def plot_rh_vs_temp(df):

    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style="whitegrid")
    sns.set_context("paper")
    sns.set_palette('bright')

    def percentile(n):
        def percentile_(x):
            return np.percentile(x, n)
        percentile_.__name__ = 'percentile_%s' % n
        return percentile_

    # Groups all grid cells of an ensemble member for a country, GWL (and implicitly RCP)
    tmp = df.groupby(['Country', 'Ensemble_Member', 'GWL', 'RCP']).agg({
        'Temperature': 'median',
        'RH': 'median',
        'FFDI': 'median',
        'Precipitation': 'median',
        'Wind_Speed': 'median',
        'Soil_Moisture': 'median' # percentile(90)
    })
    tmp.reset_index(inplace=True)

    # sizes = [23, 27, 33]
    # marker_size = pd.cut(tmp['FFDI'], [21, 25, 29, 37], labels=sizes)

    g = sns.relplot(data=tmp, x='Temperature', y='RH', hue='GWL',  # size=marker_size, # size='FFDI',
        size='FFDI', sizes=(1,200), style='RCP', col='Country', alpha=.5, kind='scatter')
    [plt.setp(ax.texts, text="") for ax in g.axes.flat]
    g.set_titles(col_template='{col_name}')
    g.set_axis_labels(x_var="Temperature ($^\circ$C)", y_var="RH (%)")
    g.savefig('/data/users/hadhy/ESMS/fire_paper/plots/temp_vs_rh_scatter.png')

    g = sns.relplot(data=tmp, x='Temperature', y='Precipitation', hue='GWL',  # size=marker_size, # size='FFDI',
        size='FFDI', sizes=(1,200), style='RCP', col='Country', alpha=.5, kind='scatter')
    [plt.setp(ax.texts, text="") for ax in g.axes.flat]
    g.set_titles(col_template='{col_name}')
    g.set_axis_labels(x_var="Temperature ($^\circ$C)", y_var="Precipitation (mm.day$^{-1}$)")
    g.savefig('/data/users/hadhy/ESMS/fire_paper/plots/temp_vs_precip_scatter.png')

    g = sns.relplot(data=tmp, x='Temperature', y='Soil_Moisture', hue='GWL',  # size=marker_size, # size='FFDI',
        size='FFDI', sizes=(1,200), style='RCP', col='Country', alpha=.5, kind='scatter')
    [plt.setp(ax.texts, text="") for ax in g.axes.flat]
    g.set_titles(col_template='{col_name}')
    g.set_axis_labels(x_var="Temperature ($^\circ$C)", y_var="Soil Moisture (kg.m$^{-2}$)")
    g.savefig('/data/users/hadhy/ESMS/fire_paper/plots/temp_vs_soilmoisture_scatter.png')

    g = sns.relplot(data=tmp, x='Temperature', y='Wind_Speed', hue='GWL',  # size=marker_size, # size='FFDI',
        size='FFDI', sizes=(1,200), style='RCP', col='Country', alpha=.5, kind='scatter')
    [plt.setp(ax.texts, text="") for ax in g.axes.flat]
    g.set_titles(col_template='{col_name}')
    g.set_axis_labels(x_var="Temperature ($^\circ$C)",  y_var="Wind Speed (m.s$^{-1}$)")
    g.savefig('/data/users/hadhy/ESMS/fire_paper/plots/temp_vs_wind_scatter.png')


def plot_cell_climatologies(df):

    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style="whitegrid")
    sns.set_context("paper")
    sns.set_palette('bright')

    # With stripplot ...
    for var in ['FFDI', 'RH', 'Temperature', 'Precipitation', 'Wind_Speed', 'Soil_Moisture']:
        print(var)
        sns_plot = sns.catplot(x='GWL', y=var, hue='RCP', data=df, col='Country', kind='box', whis=[0, 100])
        sns_plot = sns_plot.map(sns.stripplot, 'GWL', var, 'RCP', dodge=True,
                                palette={'Baseline': 'black', 'rcp2.6': 'Black', 'rcp8.5': 'black'}, size=1,
                                jitter=0.35, alpha=0.2)
        [plt.setp(ax.texts, text="") for ax in sns_plot.axes.flat]
        sns_plot.set_titles(col_template='{col_name}')
        sns_plot.savefig('/data/users/hadhy/ESMS/fire_paper/plots/figure_'+var+'.png')

    # Violin ...
    for var in ['FFDI', 'RH', 'Temperature', 'Precipitation', 'Wind_Speed', 'Soil_Moisture']:
        print(var)
        sns_plot = sns.catplot(x='GWL', y=var, hue='RCP', data=df, col='Country', kind='violin', dodge=True)
        [plt.setp(ax.texts, text="") for ax in sns_plot.axes.flat]
        sns_plot.set_titles(col_template='{col_name}')
        sns_plot.savefig('/data/users/hadhy/ESMS/fire_paper/plots/figure_'+var+'_violin.png')

    # Without stripplot ...
    for var in ['FFDI', 'RH', 'Temperature', 'Precipitation', 'Wind_Speed', 'Soil_Moisture']:
        print(var)
        sns_plot = sns.catplot(x='GWL', y=var, hue='RCP', data=df, col='Country', kind='box', whis=[0, 100])
        [plt.setp(ax.texts, text="") for ax in sns_plot.axes.flat]
        sns_plot.set_titles(col_template='{col_name}')
        sns_plot.savefig('/data/users/hadhy/ESMS/fire_paper/plots/figure_'+var+'_nopoints.png')

    # change df to long format
    df.columns = ['CellX', 'CellY', 'Ensemble Member', 'Country', 'GWL', 'RCP', 'FFDI',
                  'Relative Humidity', 'Temperature', 'Precipitation', 'Wind Speed', 'Soil Moisture']
    dflong = pd.melt(df, id_vars=['CellX', 'CellY', 'Ensemble Member', 'Country', 'GWL', 'RCP'], value_vars=['FFDI',
       'Relative Humidity', 'Temperature', 'Precipitation', 'Wind Speed', 'Soil Moisture'])

    # All in one figure (points and boxplots)
    g = sns.FacetGrid(data=dflong, col='Country', row='variable', margin_titles=True, sharex=True, sharey='row')
    g.map(sns.boxplot, 'GWL', 'value', 'RCP', hue_order=['Baseline','rcp2.6','rcp8.5'], whis=[0,100], palette='bright')
    g.map(sns.stripplot, 'GWL', 'value', 'RCP', dodge=True, hue_order=['Baseline','rcp2.6','rcp8.5'],
                                palette={'Baseline': 'black', 'rcp2.6': 'Black', 'rcp8.5': 'black'}, size=0.5,
                                jitter=0.35, alpha=0.2)
    [plt.setp(ax.texts, text="") for ax in g.axes.flat]
    g.set_titles(row_template='{row_name}', col_template='{col_name}')
    g.fig.subplots_adjust(wspace=.02, hspace=.02)
    g.savefig('/data/users/hadhy/ESMS/fire_paper/plots/figure_allvars_withpoints.png')

    # All in one figure (boxplots only)
    p = sns.FacetGrid(data=dflong, col='Country', row='variable', margin_titles=True, sharex=True, sharey='row')
    p.map(sns.boxplot, 'GWL', 'value', 'RCP', hue_order=['Baseline', 'rcp2.6', 'rcp8.5'], whis=[0, 100],
          palette=sns.color_palette(['#99a3a4','#0097ff','#d7301f'])) # palette='bright'
    [plt.setp(ax.texts, text="") for ax in p.axes.flat]
    p.set_titles(row_template='{row_name}', col_template='{col_name}')
    ylabels = ["FFDI", "Relative Humidity (%)", "Temperature ($^\circ$C)", "Precipitation (mm.day$^{-1}$)", "Wind Speed (m.s$^{-1}$)", "Soil Moisture (kg.m$^{-2}$)"]
    for i in range(len(ylabels)):
        p.axes[i, 0].set_ylabel(ylabels[i])
    # p.add_legend(loc='center right', bbox_to_anchor=(0.87, 0.47), frameon=True)
    p.add_legend(loc='center left', bbox_to_anchor=(0.76, 0.47), frameon=True)
    p.fig.subplots_adjust(wspace=0, hspace=0)
    p.savefig('/data/users/hadhy/ESMS/fire_paper/plots/figure_allvars_nopoints.png')

    # All in one figure (boxplots only), RCP x-axis, GWL colours
    p = sns.FacetGrid(data=dflong, col='Country', row='variable', margin_titles=True, sharex=True, sharey='row')
    p.map(sns.boxplot, 'RCP', 'value', 'GWL', hue_order=['Baseline', 'GWL1.5', 'GWL2', 'GWL4'], whis=[0, 100],
          palette=sns.color_palette(['#99a3a4','#0097ff','#fc641a','#d7301f'])) # palette='bright'
    [plt.setp(ax.texts, text="") for ax in p.axes.flat]
    p.set_titles(row_template='{row_name}', col_template='{col_name}')
    ylabels = ["FFDI", "Relative Humidity (%)", "Temperature ($^\circ$C)", "Precipitation (mm.day$^{-1}$)", "Wind Speed (m.s$^{-1}$)", "Soil Moisture (kg.m$^{-2}$)"]
    for i in range(len(ylabels)):
        p.axes[i, 0].set_ylabel(ylabels[i])
    # p.add_legend(loc='center right', bbox_to_anchor=(0.87, 0.47), frameon=True)
    p.add_legend(loc='center left', bbox_to_anchor=(0.76, 0.47), frameon=True)
    p.fig.subplots_adjust(wspace=0, hspace=0)
    p.savefig('/data/users/hadhy/ESMS/fire_paper/plots/figure_allvars_nopoints_RCP-xaxis.png')


def main_old():
    # This code produced plots with outliers in the variable relationships.
    # I think it was due to problems with the masking. I only looked at grid cells where FFDI>24 was recorded for a
    # certain number of days. This may have introduced some erros

    # Get the data from MASS. Possibly a problem withe the joined data, so using the above data instead
    if not os.path.isdir("/scratch/hadhy/fire/ffdi_output_joined/rcp2_6/"):
        os.makedirs("/scratch/hadhy/fire/ffdi_output_joined/rcp2_6/")
    if not os.path.isdir("/scratch/hadhy/fire/ffdi_output_joined/rcp8_5/"):
        os.makedirs("/scratch/hadhy/fire/ffdi_output_joined/rcp8_5/")

    subprocess.run(["moo", "get", "-v", "moose:/adhoc/users/inika.taylor/fire/rcp2_6_joined/*.nc", "/scratch/hadhy/fire/ffdi_output_joined/rcp2_6/"])
    subprocess.run(["moo", "get", "-v", "moose:/adhoc/users/inika.taylor/fire/rcp8_5_joined/*.nc", "/scratch/hadhy/fire/ffdi_output_joined/rcp8_5/"])

    # Getting FFDI input variables from MASS
    if not os.path.isdir("/scratch/hadhy/fire/ffdi_input_variables/rcp2_6/"):
        os.makedirs("/scratch/hadhy/fire/ffdi_input_variables/rcp2_6/")
    if not os.path.isdir("/scratch/hadhy/fire/ffdi_input_variables/rcp8_5/"):
        os.makedirs("/scratch/hadhy/fire/ffdi_input_variables/rcp8_5/")

    subprocess.run(["moo", "get", "-v", "moose:/adhoc/users/inika.taylor/fire/rcp2_6/ffdi_input_variables*.nc", "/scratch/hadhy/fire/ffdi_input_variables/rcp2_6/"])
    subprocess.run(["moo", "get", "-v", "moose:/adhoc/users/inika.taylor/fire/rcp8_5/ffdi_input_variables*.nc", "/scratch/hadhy/fire/ffdi_input_variables/rcp8_5/"])

    # Load the ensemble jobids and GWLs
    ensdf = get_ens_df()
    plot_gwl_year(ensdf)

    # Baseline start year
    bl_start_yr = 1986

    # Get the baseline data to plot
    csv_bl = '/data/users/hadhy/ESMS/fire_paper/data/odf_113_Baseline.csv'
    if not os.path.isfile(csv_bl):
        df2p_baseline = get_data_to_plot('Baseline', bl_start_yr, ensdf)
    else:
        df2p_baseline = pd.read_csv(csv_bl)
    df2p_base = df2p_baseline[df2p_baseline.RCP == 'rcp2.6'].copy()
    df2p_base['RCP'] = 'Baseline'

    # Get GWL1.5 data to plot
    csv_gwl1p5 = '/data/users/hadhy/ESMS/fire_paper/data/odf_113_GWL1.5.csv'
    if not os.path.isfile(csv_gwl1p5):
        df2p_gwl1p5 = get_data_to_plot('GWL1.5', bl_start_yr, ensdf)
    else:
        df2p_gwl1p5 = pd.read_csv(csv_gwl1p5)

    # Get GWL2 data to plot
    csv_gwl2 = '/data/users/hadhy/ESMS/fire_paper/data/odf_113_GWL2.csv'
    if not os.path.isfile(csv_gwl2):
        df2p_gwl2 = get_data_to_plot('GWL2', bl_start_yr, ensdf)
    else:
        df2p_gwl2 = pd.read_csv(csv_gwl2)

    # Get GWL4 data to plot
    csv_gwl4 = '/data/users/hadhy/ESMS/fire_paper/data/odf_113_GWL4.csv'
    if not os.path.isfile(csv_gwl4):
        df2p_gwl4 = get_data_to_plot('GWL4', bl_start_yr, ensdf)
    else:
        df2p_gwl4 = pd.read_csv(csv_gwl4)

    df = pd.concat([df2p_base, df2p_gwl1p5, df2p_gwl2, df2p_gwl4], ignore_index=True)
    plot_cell_climatologies(df)

    df = pd.concat([df2p_baseline, df2p_gwl1p5, df2p_gwl2, df2p_gwl4], ignore_index=True)
    plot_rh_vs_temp(df)


def main():

    # Get the data from MASS. This is the actual data that we will use ...
    if not os.path.isdir("/scratch/hadhy/fire/ffdi_output/rcp2_6/"):
        os.makedirs("/scratch/hadhy/fire/ffdi_output/rcp2_6/")
    if not os.path.isdir("/scratch/hadhy/fire/ffdi_output/rcp8_5/"):
        os.makedirs("/scratch/hadhy/fire/ffdi_output/rcp8_5/")

    if len(os.listdir('/scratch/hadhy/fire/ffdi_output/rcp2_6/')) < 228:
        subprocess.run(["moo", "get", "-v", "moose:/adhoc/users/inika.taylor/fire/rcp2_6/ffdi_outputs/*.nc", "/scratch/hadhy/fire/ffdi_output/rcp2_6/"])
    if len(os.listdir('/scratch/hadhy/fire/ffdi_output/rcp8_5/')) < 228:
        subprocess.run(["moo", "get", "-v", "moose:/adhoc/users/inika.taylor/fire/rcp8_5/ffdi_outputs/*.nc", "/scratch/hadhy/fire/ffdi_output/rcp8_5/"])

    # Getting FFDI input variables from MASS
    if not os.path.isdir("/scratch/hadhy/fire/ffdi_input_variables/rcp2_6/"):
        os.makedirs("/scratch/hadhy/fire/ffdi_input_variables/rcp2_6/")
    if not os.path.isdir("/scratch/hadhy/fire/ffdi_input_variables/rcp8_5/"):
        os.makedirs("/scratch/hadhy/fire/ffdi_input_variables/rcp8_5/")

    if len(os.listdir('/scratch/hadhy/fire/ffdi_input_variables/rcp2_6/')) < 798:
        subprocess.run(["moo", "get", "-v", "moose:/adhoc/users/inika.taylor/fire/rcp2_6/ffdi_input_variables*.nc", "/scratch/hadhy/fire/ffdi_input_variables/rcp2_6/"])
    if len(os.listdir('/scratch/hadhy/fire/ffdi_input_variables/rcp8_5/')) < 798:
        subprocess.run(["moo", "get", "-v", "moose:/adhoc/users/inika.taylor/fire/rcp8_5/ffdi_input_variables*.nc", "/scratch/hadhy/fire/ffdi_input_variables/rcp8_5/"])

    # Load the ensemble jobids and GWLs
    ensdf = get_ens_df()
    # plot_gwl_year(ensdf)

    # Get all data to plot
    df_12_24 = get_variables_by_ffdi_theshold(ensdf, 12, 24, ofile='/data/users/hadhy/ESMS/fire_paper/data/odf_bigdata_3countries_12_to_24.csv')
    # df_12_24.to_csv('/data/users/hadhy/ESMS/fire_paper/data/odf_bigdata_3countries_12_to_24.csv')
    df_gt_24 = get_variables_by_ffdi_theshold(ensdf, 24, np.inf, ofile='/data/users/hadhy/ESMS/fire_paper/data/odf_bigdata_3countries_gt_24.csv')
    # df_gt_24.to_csv('/data/users/hadhy/ESMS/fire_paper/data/odf_bigdata_3countries_gt_24.csv')


if __name__ == '__main__':
    main()
