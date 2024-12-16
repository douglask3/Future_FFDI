import iris
import iris.coord_categorisation
import numpy as np
import matplotlib.pyplot as plt

import iris.plot as iplt
import iris.quickplot as qplt
from iris.util import rolling_window
from pdb import set_trace

from datetime import datetime, timedelta
import os
import pandas as pd 
from libs.constrain_cubes_standard import *


def get_characters_after_string(input_string, search_string, num_characters):
    index = input_string.find(search_string)
    if index != -1:  # Check if the search string is found in the input string
        return input_string[index + len(search_string):index + len(search_string) + num_characters]
    else:
        return "Search string not found in input string"

def calculate_time(rcp, member, hmember, GWL, nyear = 20, country = '', 
                   bare_mask = '/project/climate_impacts/durban/wp3_results/fire/SAGE_bare_soil_ice_mask_regrid_gt50.pp'):
    
    out_file = 'temp/time_in_catigory_7' + country + '_' + rcp + GWL + member + str(nyear) + '.csv'
    
    print(out_file)
    
    if os.path.isfile(out_file):
        out = np.genfromtxt(out_file, delimiter=',')
        return np.transpose(out[1:,1:])
    
    directory = ffdis_dir + rcp + '/'
    start_fname = 'ffdi_' + rcp + '_' + member + '_'
    file_list = [f for f in os.listdir(directory) if f.startswith(start_fname) and f.endswith('.nc')]
    
    cubelist = iris.cube.CubeList()
    for file_name in file_list:
        cubelist.extend(iris.load(os.path.join(directory, file_name)))
    cubelist.extend(iris.load(directory + 'ffdi_' + rcp + '_' + hmember + '_1970s_2000s.nc'))
    
    ffdi = cubelist.concatenate_cube()
    
    #ffdi_file = ffdis_dir + rcp '/ffdi_joined_' + rcp + '_' + member + '.nc'
    #ffdi = iris.load_cube(ffdi_file)
    
    bare_mask = iris.load_cube(bare_mask)
    if country != '':
        ffdi = constrain_natural_earth(ffdi, country)
        bare_mask = constrain_natural_earth(bare_mask, country)
    iris.coord_categorisation.add_day_of_year(ffdi, 'time')
    print(start_fname)
    GWL_file = GWL_dir + rcp + '/' + rcp + 'gwl_' + GWL + '.txt'
    year0 = [1986, 2005]
    if GWL == 'Baseline':
        year = None 
    elif GWL == 'None':
        year = ', 2004, 2023'
    else:
        gwls = open(GWL_file).read()
        year = get_characters_after_string(gwls, member, 12)
        
    
    if year is not None and 'nan,' in year:
        out = np.empty((len(ffdi_ranges)))
        out[:] = np.nan
    else:
        
        cube_years = ffdi.coord('year').points

        def extra_dates(yeari):       
            index = np.where((cube_years >= yeari[0]) & (cube_years <= yeari[1]))[0]
            cube  = ffdi[index].copy()
            return(cube)
        
        cube0 = extra_dates(year0)
        if year is None: 
            cube = None
        else:
            year = [float(year[2:6]), float(year[8:12])]
            cube  = extra_dates(year)
        
        #clim = cube.aggregated_by('day_of_year', iris.analysis.MEAN)
        #clim0 = cube0.aggregated_by('day_of_year', iris.analysis.MEAN).data
        
        def no_days_at_ffdi(ffdi_range, cube0, cube, bare_mask):
            #mask = np.logical_or(cube.data < ffdi_range[0], cube.data > ffdi_range[1])
            
            def mean_exceedance(icube):
                r = icube.copy() 
                r.data = r.data > ffdi_range
                r = r.collapsed('time', iris.analysis.MEAN)
                r.data = r.data * 365
                return(r)
            
            exceed0 = mean_exceedance(cube0)
            if cube is None: 
                diff = exceed0
            else:
                exceed  = mean_exceedance(cube)
                diff = exceed - exceed0
            diff.data[~bare_mask.data.mask] = np.nan
            diff.data.mask[np.isnan(diff.data)] = True
            #masked_cube = clim.copy(data=np.ma.masked_where(~mask, clim.data))
            try:
                diff.coord("longitude").guess_bounds()
            except:
                pass
            try:
                diff.coord("latitude").guess_bounds()
            except:
                pass
            area_weights = iris.analysis.cartography.area_weights(diff)
            
            return  diff.collapsed(['longitude', 'latitude'], iris.analysis.MEAN,  weights = area_weights).data
            
        out = [no_days_at_ffdi(ffdi_range, cube0, cube, bare_mask) \
                    for ffdi_range in ffdi_ranges]
        out = np.array(out) 
        #out = out/np.sum(out)
        
    df = pd.DataFrame(out)
    df.to_csv(out_file)
    out = out[None, :]
    
    return out

def for_region(country):
    out_all = []
    colnames = []
    for sc in scenario_name:
        out_sc = []
        for gwl in global_warming_level:
            outi = np.array([calculate_time(sc, member, hmember, gwl, country = country)[0,:] \
                             for member, hmember in zip(members[sc], hist_members)])
            
            out_all.append(np.nanpercentile(outi, np.array([10, 50, 90]), axis = 0))
            
            #out.append(np.transpose(np.percentile(outi, np.array([10, 50, 90]), axis = 0)))
            cname = sc + '-' + gwl + '-'
            colnames.append(cname + '10')
            colnames.append(cname + '50')
            colnames.append(cname + '90')
        
        #out = [np.percentile(out_sc[0], np.array([10, 50, 90]), axis = 0)]
        #for i in  range(1,len(global_warming_level)):
        #   out.append(np.percentile(out_sc[i] - out_sc[0], np.array([10, 50, 90]), axis = 0))
        #set_trace()
        #out_all.append(np.concatenate(out, axis=0))

    out = np.transpose(np.concatenate(out_all, axis=0))
    #np.savetxt('outputs/cal_time.csv', out, 
    #            delimiter=',',
    #            header=','.join(colnames), comments='', fmt='%0.8f')
    
    rownames = ['>' + str(x) for x in ffdi_ranges]
    df = pd.DataFrame(out, index=rownames, columns=colnames)
    df = df.to_csv('outputs/cal_time-' + country + '.csv')
    df = pd.DataFrame(np.transpose(np.round(out, 3)), columns = rownames, index = colnames)
    df = combine_ranges(df)
    return df
    df = df.to_csv('outputs/cal_time-tidy' + country + '.csv')

def combine_ranges(df):
    N = 9  # Adjust this value as needed

    # Extracting the first N digits from the existing row names
    rnames = df.iloc[::3].index

    # Create an empty DataFrame to store the combined rows along with existing columns
    columns_to_preserve = df.columns.tolist()  # Get existing column names
    #columns_to_preserve.remove('YourColumnName')  # Remove the column you want to combine
    combined_df = pd.DataFrame(columns=df.columns)

    # Iterate through the rows of the original DataFrame in steps of three
    for i in range(0, len(df), 3):
        # Combine the values in each column for the group of three rows
        combined_values = df.iloc[i:i+3].apply(lambda x: f"{x.iloc[0]} ({x.iloc[1]} - {x.iloc[2]})")

        # Append the combined values to the new DataFrame
        combined_df = combined_df.append(combined_values, ignore_index=True)
    combined_df.index = rnames
    
    return combined_df
    

    
    

if __name__=="__main__":
    dir = "/scratch/dkelley/future_ffdi/data/"

    ffdis_dir = '/scratch/hadhy/fire/ffdi_output/'
    GWL_dir = '/scratch/dkelley/future_ffdi/data//GlobalWarmingLevels/'

    hist_members =['akowa', 'akowb', 'akowc', 'akowe', 'akowf', 'akowh', 'akowi', 'akowj', 'akowk', 'akowl',
                 'akowm', 'akown', 'akowo', 'akowp', 'akowq', 'akoxa', 'akoxb', 'akoxc', 'akoxd', 'akoxe',
                 'akoxg', 'akoxi', 'akoxj', 'akoxl', 'akoxm', 'akoxn', 'akoxo', 'akoxp', 'akoxq', 'akoya',
                 'akoyb', 'akoyc', 'akoye', 'akoyf', 'akoyg', 'akoyh', 'akoyi', 'akoyj', 'akoyk', 'akoyl',
                 'akoym', 'akoyn', 'akoyq', 'akoza', 'akozb', 'akozc', 'akozd', 'akoze', 'akozg', 'akozh',
                 'akozi', 'akozj', 'akozl', 'akozn', 'akozo', 'akozp', 'akozq']

    members = {'rcp2_6': ['akuka', 'akukb', 'akukc', 'akuke', 'akukf', 'akukh', 'akuki', 
                          'akukj', 'akukk', 'akukl', 'akukm', 'akukn', 'akuko', 'akukp', 
                          'akukq', 'akula', 'akulb', 'akulc', 'akuld', 'akule', 'akulg', 
                          'akuli', 'akulj', 'akull', 'akulm', 'akuln', 'akulo', 'akulp', 
                          'akulq', 'akuma', 'akumb', 'akumc', 'akume', 'akumf', 'akumg', 
                          'akumh', 'akumi', 'akumj', 'akumk', 'akuml', 'akumm', 'akumn', 
                          'akumq', 'akuna', 'akunb', 'akunc', 'akund', 'akune', 'akung', 
                          'akunh', 'akuni', 'akunj', 'akunl', 'akunn', 'akuno', 'akunp', 
                          'akunq'],
               'rcp8_5': ['aldpa', 'aldpb', 'aldpc', 'aldpe', 'aldpf', 'aldph', 'aldpi', 
                          'aldpj', 'aldpk', 'aldpl', 'aldpm', 'aldpn', 'aldpo', 'aldpp', 
                          'aldpq', 'aldqa', 'aldqb', 'aldqc', 'aldqd', 'aldqe', 'aldqg', 
                          'aldqi', 'aldqj', 'aldql', 'aldqm', 'aldqn', 'aldqo', 'aldqp', 
                          'aldqq', 'aldra', 'aldrb', 'aldrc', 'aldre', 'aldrf', 'aldrg', 
                          'aldrh', 'aldri', 'aldrj', 'aldrk', 'aldrl', 'aldrm', 'aldrn', 
                          'aldrq', 'aldsa', 'aldsb', 'aldsc', 'aldsd', 'aldse', 'aldsg',
                          'aldsh', 'aldsi', 'aldsj', 'aldsl', 'aldsn', 'aldso', 'aldsp', 
                          'aldsq']}
    
    scenario_name = ['rcp2_6', 'rcp8_5']
    global_warming_level = ['Baseline', 'None', '1_5', '2_deg', '4_deg']
    ffdi_ranges = [12.0, 24.0, 50.0, 75.0, 100.0]
    countries = ['', 'Australia', 'Brazil', 'United States of America']

    #for_region(countries)
    dfs = [for_region(country) for country in countries]
    
    first_rows = [df.iloc[0] for df in dfs]

    # Concatenating first rows into a new dataframe
    result_df = pd.concat(first_rows, axis = 1)

# Extracting subsequent rows and concatenating them in a cyclic manner
    for i, df in enumerate(dfs):
        rows = [df.iloc[j] for j in range(1, len(df))]
        cyclic_rows = rows[i:] + rows[:i]  # Shifting rows cyclically
        set_trace()
        result_df = pd.concat([result_df] + cyclic_rows)
    
# Resetting index of the result dataframe
    #result_df.reset_index(drop=True, inplace=True)
    result_df = result_df.transpose()
    # Update the index of the combined DataFrame with the extracted row name prefixes
    
    df = result_df.to_csv('outputs/cal_time-tidy.csv')
    #ffdi_ranges = [[0.0, 4.0], [4.0, 15.0], [15.0, 9999999.0]]
    #for_region(countries)
    #[for_region(country) for country in countries]
