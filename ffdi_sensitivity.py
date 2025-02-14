import os, sys
import iris
import glob
import pandas as pd
import numpy as np
import numpy.ma as ma
import cftime
from iris.util import unify_time_units
sys.path.append('/home/h02/hadhy/GitHub/wcssp_casestudies')
import std_functions as sf
sys.path.append('/home/h02/hadhy/PycharmProjects/impactstoolbox')
import indices.calc_fire as fire
import subprocess
import ffdi_variables_analysis as fva

def dummy_data():
    '''
    Calculate FFDI with simple distributions of input variables
    :return:
    '''
    import numpy as np
    from scipy.stats import skewnorm
    r = skewnorm.rvs(4, size=1000)
    tasmax = np.random.normal(30, 5, 100000)
    hurs = np.random.normal(30, 5, 100000)
    # mcarthur = fire.calc_mcarthur_index_cube(daily_max_temperature_in_degC_cube=tasmax,
    #                                          daily_min_relative_humidity_in_percent_cube=hurs,
    #                                          daily_mean_wind_speed_in_m_per_s_cube=wspd,
    #                                          daily_rainfall_in_mm_cube=pr,
    #                                          soil_moisture_in_mm_cube=mrso,
    #                                          parallel=True, processes=mp.cpu_count()
    #                                          )

def main():
    fva.get_ens_df()


if __name__ == '__main__':
    main()
