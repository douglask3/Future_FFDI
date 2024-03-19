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

def calculate_time(rcp, member, hmember, GWL, nyear = 20, country = ''):
    
    out_file = 'temp/time_in_catigory_4' + country + '_' + rcp + GWL + member + str(nyear) + '.csv'
    
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
    #set_trace()
    ffdi = cubelist.concatenate_cube()
    #ffdi_file = ffdis_dir + rcp '/ffdi_joined_' + rcp + '_' + member + '.nc'
    #ffdi = iris.load_cube(ffdi_file)
    if country != '':
        ffdi = constrain_natural_earth(ffdi, country)
    print(start_fname)
    GWL_file = GWL_dir + rcp + '/' + rcp + 'gwl_' + GWL + '.txt'
    
    if GWL == 'None':
        year = '2005'
    else:
        gwls = open(GWL_file).read()
        year = get_characters_after_string(gwls, member, 6)[2:]
    if year == 'nan,':
        out = np.empty((len(ffdi_ranges)))
        out[:] = np.nan
    else:
        cube_years = ffdi.coord('year').points

        index = np.where((cube_years > float(year)- nyear - 1) & \
                         (cube_years < float(year)))[0]
        
        iris.coord_categorisation.add_day_of_year(ffdi, 'time')
        cube  = ffdi[index].copy()
        
        #clim = cube.aggregated_by('day_of_year', iris.analysis.MEAN)
        #clim0 = cube0.aggregated_by('day_of_year', iris.analysis.MEAN).data
        
        def no_days_at_ffdi(ffdi_range, clim):
            mask = np.logical_or(cube.data < ffdi_range[0], cube.data > ffdi_range[1])
            #masked_cube = clim.copy(data=np.ma.masked_where(~mask, clim.data))
            try:
                cube.coord("longitude").guess_bounds()
            except:
                pass
            try:
                cube.coord("latitude").guess_bounds()
            except:
                pass
            area_weights = iris.analysis.cartography.area_weights(cube)
            
            return np.sum(area_weights[~mask])
            
            
        out = [no_days_at_ffdi(ffdi_range, cube) for ffdi_range in ffdi_ranges]
        

        out = out/np.sum(out)
        
        
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
            outi = outi*365
            
            out_sc.append(outi)
            
            #out.append(np.transpose(np.percentile(outi, np.array([10, 50, 90]), axis = 0)))
            cname = sc + '-' + gwl + '-'
            colnames.append(cname + '10')
            colnames.append(cname + '50')
            colnames.append(cname + '90')
        
        out = [np.percentile(out_sc[0], np.array([10, 50, 90]), axis = 0)]
        for i in  range(1,len(global_warming_level)):
           out.append(np.percentile(out_sc[i] - out_sc[0], np.array([10, 50, 90]), axis = 0))
        set_trace()
        out_all.append(np.concatenate(out, axis=0))

    set_trace()
    #np.savetxt('outputs/cal_time.csv', out, delimiter=',',
     #          header=','.join(colnames), comments='', fmt='%0.8f')
    rownames = [str(x[0]) + '-' + str(x[1]) for x in ffdi_ranges]
    df = pd.DataFrame(out, index=rownames, columns=colnames)
    df = df.to_csv('outputs/cal_time-' + country + '.csv')
    

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
    global_warming_level = ['None', '1_5', '2_deg', '4_deg']
    ffdi_ranges = [[0.0, 12.0], [12.0, 24.0], [24.0, 50.0],[50.0, 75.0], [75.0, 100.0], [100.0, 9999999.0]]
    countries = ['', 'Australia', 'Brazil', 'United States of America']
    #for_region(countries)
    [for_region(country) for country in countries]

    #ffdi_ranges = [[0.0, 4.0], [4.0, 15.0], [15.0, 9999999.0]]
    #for_region(countries)
    #[for_region(country) for country in countries]
