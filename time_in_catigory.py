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

def hours_since_1970_to_decimal_years(hours_since_1970_array):
    # Convert hours to days
    days_since_1970_array = hours_since_1970_array / 24.0

    # Calculate the datetime for 1970-01-01 00:00:00
    start_date = datetime(1970, 1, 1)

    # Add the number of days since 1970-01-01 to the start date for each element
    target_dates = start_date + np.array([timedelta(days=d) for d in days_since_1970_array])

    # Calculate the decimal years for each element
    decimal_years_array = np.array([target_date.year + (target_date.timetuple().tm_yday - 1) / 365.24 
                                    for target_date in target_dates])

    return decimal_years_array

def get_characters_after_string(input_string, search_string, num_characters):
    index = input_string.find(search_string)
    if index != -1:  # Check if the search string is found in the input string
        return input_string[index + len(search_string):index + len(search_string) + num_characters]
    else:
        return "Search string not found in input string"

def calculate_time(rcp, member, GWL, nyear = 20):

    out_file = 'temp/time_in_catigory_' + rcp + GWL + member + str(nyear) + '.csv'

    if os.path.isfile(out_file):
        out = np.genfromtxt(out_file, delimiter=',')
        return np.transpose(out[1:,1:])
    ffdi_file = ffdis_dir + rcp + '_joined' +'/ffdi_joined_' + rcp + '_' + member + '.nc'
    ffdi = iris.load_cube(ffdi_file)
    
    GWL_file = GWL_dir + rcp + '/' + rcp + 'gwl_' + GWL + '.txt'
    gwls = open(GWL_file).read()
    year = get_characters_after_string(gwls, member, 6)[2:]

    cube_years = ffdi.coord('year').points

    index = np.where((cube_years > float(year)- nyear - 1) & (cube_years < float(year)))[0]
    
    cube = ffdi[index].copy()

    iris.coord_categorisation.add_day_of_year(cube, 'time')
    clim = cube.aggregated_by('day_of_year', iris.analysis.MEAN)

    def no_days_at_ffdi(ffdi_range, clim):
        mask = np.logical_or(clim.data < ffdi_range[0], clim.data > ffdi_range[1])
        #masked_cube = clim.copy(data=np.ma.masked_where(~mask, clim.data))
        try:
            clim.coord("longitude").guess_bounds()
        except:
            pass
        try:
            clim.coord("latitude").guess_bounds()
        except:
            pass
        area_weights = iris.analysis.cartography.area_weights(clim)
        return np.sum(area_weights[~mask])
        
    
    out = [no_days_at_ffdi(ffdi_range, clim) for ffdi_range in ffdi_ranges]
    #area = no_days_at_ffdi([0, 999999], clim)

    df = pd.DataFrame(out)
    df.to_csv(out_file)
    set_trace()
    return out


if __name__=="__main__":
    dir = "/scratch/dkelley/future_ffdi/data/"

    ffdis_dir = dir + '/ffdi_output/'
    GWL_dir = dir + '/GlobalWarmingLevels/'

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
    global_warming_level = ['1_5', '2_deg', '4_deg']
    ffdi_ranges = [[0.0, 12.0], [12.0, 24.0], [24.0, 50.0],[50.0, 75.0], [75.0, 100.0], [100.0, 9999999.0]]

    calculate_time(scenario_name[0], members['rcp2_6'][0], global_warming_level[0])#, ffdi_ranges[0])



