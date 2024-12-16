# -*- coding: iso-8859-1 -*-
#written by Inika Taylor June 2019
#calculating the number of times the FFDI is over the daily threshold in each 20 year period of interest - threshold exceedance

#import modules
import iris
import numpy as np
import matplotlib.pyplot as plt
from iris.analysis import Aggregator
import iris.plot as iplt
import iris.quickplot as qplt
from iris.util import rolling_window
from pdb import set_trace
import os

dir = "/scratch/dkelley/future_ffdi/data/"

indir1 = dir + '/ffdi_output/'
indir2 = dir + '/GlobalWarmingLevels/'
outdir = dir +'/threshold_exceedance/'

#time periods - baseline 1986-2005, three Global Warming Levels (1.5, 2.0, 4.0) above the preindustrial baseline calculated from the model baseline
#as per the observed warming at Present Day (1986-2005) given in the IPCC 1.5 degrees special report

#FFDI thresholds:


#Functions


def load_ensemble_member(start_yr,end_yr,infile=None):
    '''loads in data constrained by a date '''
    year_constraint = iris.Constraint(year=lambda year: start_yr <= year <= end_yr)
    cube = iris.load_cube(infile, year_constraint)
    print(cube[0].attributes)
    return cube

def load_text_file_data(file, e):
    with open(file, 'r') as inp:
        contents = inp.read()
        ensemble_date = contents.splitlines()
        ensemble_info = ensemble_date[e].split(',')
        ensemble_name = ensemble_info[0]
        print(ensemble_name)
        start_yr = ensemble_info[1].strip()
        end_yr = ensemble_info[2].strip()
        #print(start_yr)
        #print(end_yr)
        return start_yr, end_yr

# Note: in order to meet the requirements of iris.analysis.Aggregator, it must
# do the calculation over an arbitrary (given) data axis.
def count_spells(data, threshold, axis, spell_length):
    """
    Function to calculate the number of points in a sequence where the value
    has exceeded a threshold value for at least a certain number of timepoints.

    Generalised to operate on multiple time sequences arranged on a specific
    axis of a multidimensional array.

    Args:

    * data (array):
        raw data to be compared with value threshold.

    * threshold (float):
        threshold point for 'significant' datapoints.

    * axis (int):
        number of the array dimension mapping the time sequences.
        (Can also be negative, e.g. '-1' means last dimension)

    * spell_length (int):
        number of consecutive times at which value > threshold to "count".

    """
    if axis < 0:
        # just cope with negative axis numbers
        axis += data.ndim
    # Threshold the data to find the 'significant' points.
    data_hits = data > threshold
    #print('data hits', data_hits)

    # Make an array with data values "windowed" along the time axis.
    hit_windows = rolling_window(data_hits, window=spell_length, axis=axis)
    #print('hit windows', hit_windows.data)

    # Find the windows "full of True-s" (along the added 'window axis').
    full_windows = np.all(hit_windows, axis=axis+1)
    #print('full windows', full_windows.data)

    # Count points fulfilling the condition (along the time axis).
    spell_point_counts = np.sum(full_windows, axis=axis, dtype=int)

    return spell_point_counts



def main(threshold_level):
    #ensemble members for rcp2.6 future runs
    fut_rcp2_6 = ['akuka', 'akukb', 'akukc', 'akuke', 'akukf', 'akukh','akuki', 'akukj', 'akukk', 'akukl',
                'akukm', 'akukn', 'akuko', 'akukp', 'akukq', 'akula', 'akulb', 'akulc', 'akuld', 'akule',
                'akulg', 'akuli', 'akulj', 'akull', 'akulm', 'akuln', 'akulo', 'akulp', 'akulq', 'akuma',
                'akumb', 'akumc', 'akume', 'akumf', 'akumg', 'akumh', 'akumi', 'akumj', 'akumk', 'akuml',
                'akumm', 'akumn', 'akumq', 'akuna', 'akunb', 'akunc', 'akund', 'akune', 'akung', 'akunh',
                'akuni', 'akunj', 'akunl', 'akunn', 'akuno', 'akunp', 'akunq']
    #fut_rcp2_6 = ['akuka', 'akukb']
    #ensemble members for rcp8.5 future runs
    fut_rcp8_5 = ['aldpa', 'aldpb', 'aldpc', 'aldpe', 'aldpf', 'aldph', 'aldpi', 'aldpj', 'aldpk', 'aldpl',
                'aldpm', 'aldpn', 'aldpo', 'aldpp', 'aldpq', 'aldqa', 'aldqb', 'aldqc', 'aldqd', 'aldqe',
                'aldqg', 'aldqi', 'aldqj', 'aldql', 'aldqm', 'aldqn', 'aldqo', 'aldqp', 'aldqq', 'aldra',
                'aldrb', 'aldrc', 'aldre', 'aldrf', 'aldrg', 'aldrh', 'aldri', 'aldrj', 'aldrk', 'aldrl',
                'aldrm', 'aldrn', 'aldrq', 'aldsa', 'aldsb', 'aldsc', 'aldsd', 'aldse', 'aldsg', 'aldsh',
                'aldsi', 'aldsj', 'aldsl', 'aldsn', 'aldso', 'aldsp', 'aldsq']
    #fut_rcp8_5 = ['aldpa', 'aldpb']
    scenario_name = ['rcp2_6', 'rcp8_5']
    global_warming_level = ['1_5', '2_deg', '4_deg'] #global warming levels
    #read in mask for burnable land here;
    mask_dir = '/project/climate_impacts/durban/wp3_results/fire/SAGE_bare_soil_ice_mask_regrid_gt50.pp'
    #browser()
    #non_burnable_land_mask = iris.load_cube(mask_dir)

    #loop over the scenarios
    for s, scen in enumerate(scenario_name):
        print(s, scen)
        if scen == 'rcp2_6':
            ensemble_member = fut_rcp2_6
            print(ensemble_member)
        elif scen == 'rcp8_5':
            ensemble_member = fut_rcp8_5
            print(ensemble_member)
        else:
            print(s, scen)
            raise Exception('check scenario name')


        #loop over the ensemble members
        for e, ensm in enumerate(ensemble_member):
            print(e, ensm)
            #read in the daily FFDI values
            #filename = '/net/spice/scratch/hadin/fire/ffdi_output/rcp2_6/ffdi_joined/ffdi_joined_rcp2_6_akuka.nc'
            
            outfile1 = outdir + scen + '/baseline_threshold_exceedance_' + \
                        '-'.join(str(i) for i in threshold_level) + scen + '_' + \
                        '_' + ensm + '_' + '_1986_2005.nc'
            
            filename = indir1 + scen + '_joined/ffdi_joined_' + scen + '_' + ensm + '.nc'
            SPELL_COUNT = Aggregator('spell_count', count_spells,
                                     units_func=lambda units: 1)
            
            spell_years = 1
            if os.path.isfile(outfile1):  
                annual_avg_bsln = iris.load_cube(outfile1)
            else:
                print(filename)
                baseline = load_ensemble_member(1986, 2005, filename)
                print(baseline)
                print ('max:', baseline.data.max())
                #print ('min:', baseline.data.min())
                #print ('mean:', baseline.data.mean())

                #calculate the threshold exceedance in the baseline period for multiple thresholds
                # Make an aggregator from the user function.
                

                # Define the parameters of the test.
                #calc FFDI threshold exceedance - Very High (and above) 24+ as this is considered Fire Weather Season
                #look at every cell for where it is 24 or above (i.e. above 23) and count the number of days above then divide by 20
                #to get the annual average
                
                #also caluclate the number of days above the 'severe' FFDI level (50+) to look at more dangerous events
                #threshold_level = 49 #severe FFDI

            # Calculate the statistic.
                
                fire_periods_bsln = baseline.collapsed('time', SPELL_COUNT, threshold=threshold_level[0], spell_length=spell_years) - baseline.collapsed('time', SPELL_COUNT, threshold=threshold_level[1], spell_length=spell_years)
                
                fire_periods_bsln.rename('Number of days over threshold')
                print(fire_periods_bsln)
                print ('max:', fire_periods_bsln.data.max())
                #print ('min:', fire_periods_bsln.data.min())
                #print ('mean:', fire_periods_bsln.data.mean())

                #convert into an annual mean by dividing by 20 to give annual avg number of days over the threshold
                annual_avg_bsln = fire_periods_bsln / 20
                print(annual_avg_bsln)
                print ('max:', annual_avg_bsln.data.max())
                #print ('min:', annual_avg_bsln.data.min())
                #print ('mean:', annual_avg_bsln.data.mean())
                #write out the baseline threshold exceedance
                #outfile1 = outdir + scen + '/baseline_threshold_exceedance_SEVERE_' + scen + '_' + ensm + '_1986_2005.nc'
                print(outfile1)
                iris.save(annual_avg_bsln, outfile1)

                #need to apply a mask for burnable land e.g. mask out non-burnable land

            # Plot the results.
            #qplt.contourf(annual_avg_bsln, cmap='RdYlBu_r')
            #plt.gca().coastlines()
            #iplt.show()

            #loop over the three global warming levels to
            #calculate the threshold exceedance in the future
            for g, gwl in enumerate(global_warming_level):
                print(gwl)
                #read in global warming level file - text file with start and end dates of 20 year period first reaching the GWL
                #file = '/net/spice/scratch/hadin/fire/GlobalWarmingLevels/rcp2_6/rcp2_6gwl_1_5.txt'
                file = dir + '/GlobalWarmingLevels/' + scen + '/' + scen + 'gwl_' + gwl + '.txt'
                print(file)
                start_year, end_year = load_text_file_data(file, e)
                if start_year == 'nan':
                    print('here', start_year)
                    print('not reached this global warming level', gwl)
                    continue

                elif start_year != 'nan':
                    start_year = int(start_year)
                    end_year = int(end_year)
                    print('start year', start_year)
                    print('end year', end_year)
                    
                    outfile2 = outdir + scen + '/future_threshold_exceedance' + \
                        '-'.join(str(i) for i in threshold_level) + '_' + \
                         scen + '_' + ensm + '_' + gwl + '_' + str(start_year) + \
                         '_' + str(end_year) + '.nc'
                    if os.path.isfile(outfile2):  
                        annual_avg_future = iris.load_cube(outfile2)
                    else:
                        future = load_ensemble_member(start_year, end_year, filename)
                        print('future: ', future)
                        print(future.coords('year'))
                        print ('max:', future.data.max())
                        #print ('min:', future.data.min())
                        #print ('mean:', future.data.mean())

                        # Calculate the statistic.
                       
                        fire_periods_future = future.collapsed('time', SPELL_COUNT, threshold=threshold_level[0], spell_length=spell_years) - future.collapsed('time', SPELL_COUNT, threshold=threshold_level[1], spell_length=spell_years)
                        

                        fire_periods_future.rename('Number of days over threshold')
                        print(fire_periods_future)
                        print ('max:', fire_periods_future.data.max())
                        #print ('min:', fire_periods_future.data.min())
                        #print ('mean:', fire_periods_future.data.mean())
                        annual_avg_future = fire_periods_future / 20
                        #write out future threshold exceedance
                        
                        #outfile2 = outdir + scen + '/future_threshold_exceedance_SEVERE_' + scen + '_' + ensm + '_' + gwl + '_' + str(start_year) + '_' + str(end_year) + '.nc'
                        print(outfile2)
                        iris.save(annual_avg_future, outfile2)
                        #convert into an annual mean by dividing by 20 to give annual avg number of days over the threshold
                    
                    print(annual_avg_future)
                    print ('max:', annual_avg_future.data.max())
                        #print ('min:', annual_avg_future.data.min())
                    #print ('mean:', annual_avg_future.data.mean())

                    # Plot the results.
                    #qplt.contourf(annual_avg_future, cmap='RdYlBu_r')
                    #plt.gca().coastlines()
                    #iplt.show()

                    #need to apply a mask for burnable land e.g. mask out non-burnable land




if __name__ == '__main__':

    #Very High FFDI
    main(threshold_level = [4, 15])
    main(threshold_level = [0, 11])
    main(threshold_level = [12, 23])
    main(threshold_level = [23, 10000000])
