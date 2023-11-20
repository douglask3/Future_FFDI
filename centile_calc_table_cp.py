'''Feb 2020 read in anomalies of number of days above the ffdi threshold, average for a region and then calculate the 10th, 50th, 90th percentiles and write out in a table '''

#imports
import sys
sys.path.append('../MaxEnt/libs/')
from constrain_cubes_standard import *
import iris
import numpy as np
import iris.coord_categorisation
import glob
import iris.plot as iplt
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import cartopy
import numpy.ma as ma
import iris.quickplot as qpl
import ascend
from pdb import set_trace

#Functions

def country_shape(country):
    """ Load the natural earth scountry shape

    Inputs
    * country
        A string of the country name to search for

    Returns
    An ASCEND Shape object (cascaded union of all available country shapes)

    """
    shps = ascend.shape.load_shp(ascend.EXAMPLE_NATURAL_EARTH, name=country)
    if len(shps) == 0:
        raise ValueError
    return shps.cascaded_union()

def calculate_weights(shp, cube):
    """ Calculate the land area weights for each grid cell and multiply by the area overlap with a shape

    Inputs
    * shp
        ASCEND shape object
    * icube
        input cube to calculate weights for

    Returns
    Weights cube of size icube

    """
    land_area = iris.analysis.cartography.area_weights(cube)
    weights_cubes = []
    
    shp.mask_cube_inplace(cube.copy(), minimum_weight=0.1, weight_cubes=weights_cubes)
    
    weights = weights_cubes[0].data * land_area  # shape weights * area (either crop value or cartogrpahy
    if weights.shape != cube.shape:
        weights = iris.util.broadcast_to_shape(weights, cube.shape, (1, 2))
    return weights


def main():
    dir = "/scratch/dkelley/future_ffdi/"
    indir = dir + "/outputs/thresh_exceed_anomaly/"
    outdir_scratch = dir + '/outputs/thresh_exceed_anomaly/tables/'
    outdir_plots = 'figs/'#dir + '/net/home/h05/hadin/Paper_writing/Fire/new_plots/tables/'
    scenario = ['rcp2_6', 'rcp8_5']
    scenario_labels = ['RCP2.6', 'RCP8.5']
    #threshold = ['Very_High','SEVERE']
    #threshold_labels = ['Very High', 'Severe']
    threshold = ['4-15']
    threshold_labels = ['4-15']    
    global_warming_level = ['1_5', '2_deg', '4_deg']
    gwl_labels = ['1.5 degrees', '2.0 degrees', '4.0 degrees']
    region = ['Global', 'Australia', 'Brazil', 'United States']
    region_labels = ['Global', 'Australia', 'Brazil', 'United_States']
    #ensemble members for rcp2.6 future runs
    fut_rcp2_6 = ['akuka', 'akukb', 'akukc', 'akuke', 'akukf', 'akukh','akuki', 'akukj', 'akukk', 'akukl',
            'akukm', 'akukn', 'akuko', 'akukp', 'akukq', 'akula', 'akulb', 'akulc', 'akuld', 'akule',
            'akulg', 'akuli', 'akulj', 'akull', 'akulm', 'akuln', 'akulo', 'akulp', 'akulq', 'akuma',
            'akumb', 'akumc', 'akume', 'akumf', 'akumg', 'akumh', 'akumi', 'akumj', 'akumk', 'akuml',
            'akumm', 'akumn', 'akumq', 'akuna', 'akunb', 'akunc', 'akund', 'akune', 'akung', 'akunh',
            'akuni', 'akunj', 'akunl', 'akunn', 'akuno', 'akunp', 'akunq']
    #ensemble members for rcp8.5 future runs
    fut_rcp8_5 = ['aldpa', 'aldpb', 'aldpc', 'aldpe', 'aldpf', 'aldph', 'aldpi', 'aldpj', 'aldpk', 'aldpl',
            'aldpm', 'aldpn', 'aldpo', 'aldpp', 'aldpq', 'aldqa', 'aldqb', 'aldqc', 'aldqd', 'aldqe',
            'aldqg', 'aldqi', 'aldqj', 'aldql', 'aldqm', 'aldqn', 'aldqo', 'aldqp', 'aldqq', 'aldra',
            'aldrb', 'aldrc', 'aldre', 'aldrf', 'aldrg', 'aldrh', 'aldri', 'aldrj', 'aldrk', 'aldrl',
            'aldrm', 'aldrn', 'aldrq', 'aldsa', 'aldsb', 'aldsc', 'aldsd', 'aldse', 'aldsg', 'aldsh',
            'aldsi', 'aldsj', 'aldsl', 'aldsn', 'aldso', 'aldsp', 'aldsq']

    mask_dir = '/project/climate_impacts/durban/wp3_results/fire/SAGE_bare_soil_ice_mask_regrid_gt50.pp'
    non_burnable_land_mask = iris.load_cube(mask_dir)
    bare_soil_ice_mask = non_burnable_land_mask.copy()
    bare_soil_ice_mask.data = ma.where(non_burnable_land_mask.data == 1, 0, non_burnable_land_mask.data)
    bare_soil_ice_mask.data.soften_mask()
    bare_soil_ice_mask.data.mask = ma.nomask
    bare_soil_ice_mask.data = np.where(bare_soil_ice_mask.data > 1e+20, 1, bare_soil_ice_mask.data)

    #read data in - need all the ensemble members so read in based on scenario, threshold and GWL and all
    for s, scen in enumerate(scenario):

        if scen == 'rcp2_6':
            ensemble_member = fut_rcp2_6
            #print(ensemble_member)
        elif scen == 'rcp8_5':
            ensemble_member = fut_rcp8_5
            #print(ensemble_member)
        else:
            print(s, scen)
            raise Exception('check scenario name')

        for t, thresh in enumerate(threshold):

            for g, gwl in enumerate(global_warming_level):

                if g == 0:
                    x = 57
                    #print(x)
                elif g > 0:
                    #check for missing files as not all ensemble members reach each GWL

                    check_files = glob.glob(indir + scen + '/Anomaly_of_threshold_exceedance_' + scen + '_' + thresh + '_*_' + gwl + '.nc')
                    #check_files = glob.glob(indir + scen + '/Anomaly_of_threshold_exceedance_' + scen + '_' + thresh + '_*_' + gwl + 'masked.nc') # for masked data
                    #print(check_files)
                    #print('HELLLOO')
                    check_files.sort()
                    #print(check_files)

                    x = len(check_files) #for adding to the legend of the plots
                    #print('x', x)
                    #raise Exception

                if x < 57:
                    new_ensemble = []
                    for string in check_files:
                        #print('t', t)
                        #'/net/spice/scratch/hadin/fire/thresh_exceed_anomaly/Anomaly_of_threshold_exceedance_rcp2_6_SEVERE_akunp_2_deg.nc'
                        #'/net/spice/scratch/hadin/fire/thresh_exceed_anomaly/Anomaly_of_threshold_exceedance_rcp2_6_Very_High_akunp_2_deg.nc'
                        if t == 0:
                            #a = string[108:113]
                            #a = string[103:108]
                            a = string[111:116]
                            #print('a', a)
                        #if t == 1:
                        #    a = string[105:110]
                            #print('a', a)
                        new_ensemble.append(a)
                    #print(new_ensemble)
                    ensemble_model_name = new_ensemble
                else:
                    ensemble_model_name = ensemble_member

                #there is no 4 degrees for the rcp2.6 scenario so want it to just continue when that happens
                if x != 0:

                    for r, reg in enumerate(region):
                        if r == 0:
                            ensemble_list = []

                            #loop over ensemble members to read in, average and append to ensemble cube
                            for e, ensm in enumerate(ensemble_model_name):
                                region_filename = indir + scen + '/Anomaly_of_threshold_exceedance_' + scen + '_' + thresh + '_' + ensm + '_' + gwl + '.nc'
                                #region_filename = indir + scen + '/Anomaly_of_threshold_exceedance_' + scen + '_' + thresh + '_' + ensm + '_' + gwl + 'masked.nc' # for the masked data
                                print(region_filename)
                                try:
                                    region_cube = iris.load_cube(region_filename)
                                except:
                                    set_trace()
                                #print(region_cube)

                                #apply the non burnable land mask
                                mskd_region_cube = region_cube.copy()
                                mskd_region_cube.data = region_cube.data * bare_soil_ice_mask.data

                                #calculate the regional average
                                region_average = mskd_region_cube.collapsed(['latitude','longitude'], iris.analysis.MEAN)
                                #print(region_average)
                                ensemble_list.append(region_average.data)

                            print(len(ensemble_list))
                            #print(ensemble_list)
                            #convert to a numpy array
                            ensemble_array = np.asarray(ensemble_list)
                            #print(ensemble_array)
                            #print(type(ensemble_array))

                            #calculate the percentiles - 10th, 50th, 90th across the whole ensemble
                            tenth_percentile = np.percentile(ensemble_array, 10)
                            #print('10th', tenth_percentile)
                            tenth_round = round(tenth_percentile, 2)

                            fiftieth_percentile = np.percentile(ensemble_array, 50)
                            #print('50th', fiftieth_percentile)
                            fiftieth_round = round(fiftieth_percentile, 2)

                            ninetieth_percentile = np.percentile(ensemble_array, 90)
                            #print('90th', ninetieth_percentile)
                            ninetieth_round = round(ninetieth_percentile, 2)

                            #write out data for a table - one for each region, scenario and threshold (with all three GWLS included)
                            #with open(outdir_scratch + '/' + scen + '_' + thresh + '_' + gwl + '_' + reg + '_' + 'percentiles.txt', 'w') as text_file:
                            #    print("10th percentile:   {}".format(tenth_round), file=text_file)
                            #    print("50th percentile:   {}".format(fiftieth_round), file=text_file)
                            #    print("90th percentile:   {}".format(ninetieth_round), file=text_file)

                            #for the masked data (masking out non burnable land)
                            with open(outdir_scratch + '/' + scen + '_' + thresh + '_' + gwl + '_' + reg + '_' + 'percentiles_mskd.txt', 'w') as text_file:
                                print("10th percentile:   {}".format(tenth_round), file=text_file)
                                print("50th percentile:   {}".format(fiftieth_round), file=text_file)
                                print("90th percentile:   {}".format(ninetieth_round), file=text_file)

                            with open(outdir_plots + '/' + scen + '_' + thresh + '_' + gwl + '_' + reg + '_' + 'percentiles_mskd.txt', 'w') as text_file:
                                print("10th percentile:   {}".format(tenth_round), file=text_file)
                                print("50th percentile:   {}".format(fiftieth_round), file=text_file)
                                print("90th percentile:   {}".format(ninetieth_round), file=text_file)

                        if r > 0:
                            shp = country_shape(reg)
                            ensemble_list = []
                            
                            #loop over ensemble members to read in, average and append to ensemble cube
                            for e, ensm in enumerate(ensemble_model_name):
                                region_filename = indir + scen + '/Anomaly_of_threshold_exceedance_' + scen + '_' + thresh + '_' + ensm + '_' + gwl + '.nc'
                                #region_filename = indir + scen + '/Anomaly_of_threshold_exceedance_' + scen + '_' + thresh + '_' + ensm + '_' + gwl + 'masked.nc' # for the masked data
                                print(region_filename)
                                region_cube = iris.load_cube(region_filename)
                                #print(region_cube)

                                #calculate the regional average
                                # check for lon/lat bounds
                                for coord in region_cube.coord(axis='y'), region_cube.coord(axis='x'):
                                    if coord.has_bounds() is False:
                                        coord.guess_bounds()
                                region_cube = constrain_natural_earth(region_cube.copy(), reg, constrain = False)
                                reg_weights = iris.analysis.cartography.area_weights(region_cube)
                                #weights = iris.analysis.cartography.area_weights(cube)
                                #reg_weights = calculate_weights(shp, region_cube)  # get weights values
                                #print(type(bsln_weights))
                                reg_tmp = reg_weights
                                if len(reg_weights.shape) != len(region_cube.shape):
                                    if len(reg_weights.shape) == 2:  # if they are different sizes and weights is 2d
                                        reg_tmp = iris.util.broadcast_to_shape(reg_weights, region_cube.shape, (1, 2))
                                        #print('here 1')
                                    if len(reg_weights.shape) == 3:  # if they are different sizes and weights is 3d
                                        reg_tmp = iris.util.broadcast_to_shape(reg_weights[0], region_cube.shape, (1, 2))
                                        #print('here 2')
                                    else:
                                        raise ValueError

                                xname = region_cube.coord(axis="x").name()
                                yname = region_cube.coord(axis="y").name()

                                #apply the non burnable land mask
                                mskd_region_cube = region_cube.copy()
                                mskd_region_cube.data = region_cube.data * bare_soil_ice_mask.data


                                region_average = mskd_region_cube.collapsed([xname, yname], iris.analysis.MEAN, weights=reg_tmp)
                                #print(region_average)
                                ensemble_list.append(region_average.data)

                            print(len(ensemble_list))
                            #print(ensemble_list)
                            #convert to a numpy array
                            ensemble_array = np.asarray(ensemble_list)
                            #print(ensemble_array)
                            #print(type(ensemble_array))

                            #calculate the percentiles - 10th, 50th, 90th across the whole ensemble
                            tenth_percentile = np.percentile(ensemble_array, 10)
                            #print('10th', tenth_percentile)
                            tenth_round = round(tenth_percentile, 2)
                            #print('10th', tenth_round)

                            fiftieth_percentile = np.percentile(ensemble_array, 50)
                            #print('50th', fiftieth_percentile)
                            fiftieth_round = round(fiftieth_percentile, 2)

                            ninetieth_percentile = np.percentile(ensemble_array, 90)
                            #print('90th', ninetieth_percentile)
                            ninetieth_round = round(ninetieth_percentile, 2)

                            #write out data for a table - one for each region, scenario and threshold (with all three GWLS included)
                            #with open(outdir_scratch + '/' + scen + '_' + thresh + '_' + gwl + '_' + region_labels[r] + '_' + 'percentiles.txt', 'w') as text_file:
                            #    print("10th percentile:   {}".format(tenth_round), file=text_file)
                            #    print("50th percentile:   {}".format(fiftieth_round), file=text_file)
                            #    print("90th percentile:   {}".format(ninetieth_round), file=text_file)

                            #for the masked data
                            with open(outdir_scratch + '/' + scen + '_' + thresh + '_' + gwl + '_' + region_labels[r] + '_' + 'percentiles_mskd.txt', 'w') as text_file:
                                print("10th percentile:   {}".format(tenth_round), file=text_file)
                                print("50th percentile:   {}".format(fiftieth_round), file=text_file)
                                print("90th percentile:   {}".format(ninetieth_round), file=text_file)

                            with open(outdir_plots + '/' + scen + '_' + thresh + '_' + gwl + '_' + region_labels[r] + '_' + 'percentiles_mskd.txt', 'w') as text_file:
                                print("10th percentile:   {}".format(tenth_round), file=text_file)
                                print("50th percentile:   {}".format(fiftieth_round), file=text_file)
                                print("90th percentile:   {}".format(ninetieth_round), file=text_file)



if __name__ == '__main__':
    main()
