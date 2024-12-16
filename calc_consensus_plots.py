'''written by Inika Taylor Jan 2020 to create consensus plots - adapted from Chris Kents consensus code '''

#imports
import iris
import numpy as np
import iris.coord_categorisation
import glob
import iris.plot as iplt
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import cartopy
import numpy.ma
import iris.quickplot as qplt
import numpy.ma as ma
from pdb import set_trace
#Functions
class MyConsensus(object):
    def __init__(self, levels, values, names, colors):
        """ A class for setting up the consensus data and plotting

        Inputs
        * levels
            bounds for model consensus, between 0 and 1 i.e. [0.5, 0.9, 1.0]
        * values
            values to look for in a cube
        * names
            names of the associated values
        * colors
            colours of values - should be a list of len(names), each of which is len(levels-1)

        """
        self.levels = levels
        self.values = values
        self.names = names
        self.colors = colors
        self.n_levels = len(levels)
        self.n_values = len(values)

        # check the inputs match up
        if len(values) != len(names) or len(values) != len(colors):
            raise ValueError
        for x in colors:
            if len(values) != len(x):
                raise ValueError

    def plot_colorbar(self, ax=None, show=True):
        """ plot the color bar as it's own figure, unless ax is passed to it """
        cdata = np.zeros([self.n_values, self.n_levels - 1])
        ii = 0
        cmap_colors = []  # list to hold colors
        cmap_levels = [0., ]  # list to hold levels for colors

        for cols in self.colors:
            ivals = [x + 10 ** ii for x in range(len(cols))]
            cdata[ii] = ivals
            cmap_colors.extend(cols)
            cmap_levels.extend(np.array(ivals) + 0.5)
            ii += 1

        if ax is None:
            #plt.figure(figsize=(8, 1.5))
            plt.figure(figsize=(12, 3))
            plt.subplots_adjust(right=0.6, left=0.05, top=0.7, bottom=0.02)

        cmap = mc.ListedColormap(cmap_colors)
        norm = mc.BoundaryNorm(cmap_levels, cmap.N)
        plt.pcolormesh(cdata, cmap=cmap, norm=norm)
        plt.xticks(range(self.n_levels), self.levels, size=20)
        plt.yticks([], [])
        plt.tick_params(axis='x',  which='both',  top=True,  bottom=False,  labelbottom=False, labeltop=True)

        for ii, tt in enumerate(self.names):
            plt.text(self.n_values + 0.2, ii + 0.5, tt, size=16, ha="left", va='center')

        if show:
            plt.show()
            plt.close()
        else:
            return plt.gca()

    def plot_sigcube_map(self, sigcube):
        """ Plot a cube which has been converted to the correct values

        Inputs
        * sigcube
            An iris cube which contains indices of no change, positive change or negative change

        Returns
        A list of the plotting pcolormesh objects
        """
        cplist = []
        for value, colors in zip(self.values, self.colors):
            cmap = mc.ListedColormap(colors)
            norm = mc.BoundaryNorm(self.levels, cmap.N)
            pp = sigcube[0].copy()
            pp.data = (sigcube.data == value).sum(axis=0) / float(sigcube.shape[0])
            #pp.data = np.ma.masked_less(pp.data, self.levels[0])
            #print(pp.data)
            #print(pp.data.mask)           
            cp = iplt.pcolormesh(pp, cmap=cmap, norm=norm, zorder=9)
            cplist.append(cp)
        return cplist


# colors from inikas drought paper - three shades of three colours - must change these to make relevant for fire!!!
#INIKA_COLORS = [['#67c2a5', '#2c9452', '#005723'],   # decrease
#                ['#f1de87', '#d9c573', '#c1ad5e'],   # no change
#                ['#fd8e3c', '#ea2c20', '#9c0026'], ]  # increase

#INIKA_COLORS = [['#67c2a5', '#2c9452', '#005723'],   # decrease
#                ['#ffeeee', '#ddcccc', '#aa9999'],   # no change
#                ['#fd8e3c', '#ea2c20', '#9c0026'], ]  # increase

INIKA_COLORS = [['#67c2a5', '#2c9452', '#005723'],   # decrease
                ['#ffeeee', '#ddcccc', '#aa9999'],   # no change
                ['#E2A8A8', '#C65454', '#AA0000'], ]  # increase


# setup a class to hold the consensus information for processing
SETUP1 = MyConsensus([0.5, 0.7, 0.9, 1.0], [-1, 0, 1],
                     ['Decrease', 'No change', 'Increase'], INIKA_COLORS)



def calculate_ensemble_change(ensemble, byears, fyears, thresh, scen, indir, outdir, bare_soil_ice_mask):
#     """ Calculate model change - convert each model value into changes or
#     no changes - giving them a -1, 0 or 1, based on the averages for two periods. Combine these into a
#     single cube of size (nmodels, lat, lon)
#
#     Inputs
#     * models
#         A list of the model names to process
#     * baselineyears
#         Specific years to be loaded e.g. range(1981, 2011)
#     * futureyears
#         Specific years to be loaded for the future period e.g. range(2041, 2071)
#
#     Returns
#     A cube of significant and nonsignificant changes in the mean of size nmodels, lat, lon
#     """
    clist = iris.cube.CubeList()

     # loop over models in ensemble, determine if there are significant changes
    for im, ensemble_member in enumerate(ensemble):
         #bcube = get_spei_data(runid, model, rwin, month, baselineyears)
         #fcube = get_spei_data(runid, model, rwin, month, futureyears)

        #read in baseline data
        
        #/scratch/dkelley/future_ffdi/data//threshold_exceedance/rcp2_6/baseline_threshold_exceedance_12-23rcp2_6__akuli__1986_2005.nc

        baseline_filename = indir + scen + '/*' + thresh + scen + '__' + ensemble_member + '__' + byears + '*.nc'
        
        try:
            baseline_gwl_cube = iris.load_cube(baseline_filename)
        except:
            set_trace()

        #read in future data
        fut_filename = indir + scen + '/*' + thresh + '_' + scen + '_' + ensemble_member + '_' + fyears + '*.nc'
        
        fut_gwl_cube = iris.load_cube(fut_filename)

        # create a new change cube
        psig = fut_gwl_cube.copy()
        psig.add_aux_coord(iris.coords.AuxCoord(im, long_name="model"))
        psig.attributes = []
        #psig.remove_coord("time")
        #psig.remove_coord("year")
        #psig.remove_coord("month_number")

        psig.data *= 0.  # mark all as insignificant

        anomaly_cube = fut_gwl_cube - baseline_gwl_cube

        #apply the non burnable land mask
        masked_cube = anomaly_cube.copy()
        masked_cube.data = anomaly_cube.data * bare_soil_ice_mask.data

        #print(np.max(anomaly_cube.data))
        #print(np.ma.is_masked(anomaly_cube.data))
        #qplt.pcolormesh(anomaly_cube)
        #plt.show()
        #raise Exception('stop')

        #write out the anomaly data
        outfile1 = outdir + scen + '/Anomaly_of_threshold_exceedance_' + scen + '_' + thresh + '_' + ensemble_member + '_' + fyears + '.nc'
        print(outfile1)
        iris.save(anomaly_cube, outfile1)

        #write out the masked anomaly data
        outfile2 = outdir + scen + '/Anomaly_of_threshold_exceedance_' + scen + '_' + thresh + '_' + ensemble_member + '_' + fyears + 'masked.nc'
        print(outfile2)
        iris.save(masked_cube, outfile2)

        # mark increase in ffdi, mark decrease in ffdi
        #pp = np.abs(base_mu.data + sf * base_se.data)
        #psig.data[anomaly_cube.data >= 1.] = 1.

        #psig.data = np.ma.where(anomaly_cube.data > 1., 1., anomaly_cube.data)
        #psig.data[anomaly_cube.data < -1.] = -1.

        #masked dataset
        psig.data = np.ma.where(masked_cube.data > 1., 1., masked_cube.data)
        psig.data[masked_cube.data < -1.] = -1.

        clist.append(psig)
    
    
    try:
        out =  clist.merge_cube()
    except:
        set_trace()
    return out


#plot the consensus
def plot_single_ensemble_consenus(setup, ensemble, byears, fyears, title, thresh, scen, indir, outdir, plotdir, plotdir2, bare_soil_ice_mask):
    """ Plot the consensus map for a single ensemble

    Inputs
    * setup
        A myconsensus object - contains info on how to plot the data
    * ensemble
        list of the model names to process
    * runid
        string of the spei runid
    * byears
        a list of the baseline years to use
    * fyears
        a list of the future years to use

    * extent
        the wider region extent to speed up processing [x0, x1, y0, y1]
    * plot_extent
        the extent to be plotted [x0, x1, y0, y1] - make this smaller than extent to avoid annoying white bars!
    * title
        the plot title

    """
    # get ensemble data
    cc = calculate_ensemble_change(ensemble, byears, fyears, thresh, scen, indir, outdir, bare_soil_ice_mask)
    for coord in "longitude", "latitude":
        cc.coord(coord).bounds = None  # workaround for now - seemed to be affecting the mapping!

    # get region of interest
    #ccreg = cc.intersection(longitude=extent[:2], latitude=extent[2:])
    ccreg = cc #to do the whole globe hopefully

    # set up plot
    #plt.figure(figsize=(5, 4.75))
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(bottom=0.05, top=0.9, right=0.95, left=0.05)

    # plot sig cube
    setup.plot_sigcube_map(ccreg)
    ax = plt.gca()
    add_plot_features(ax)
    #ax.set_extent(plot_extent)
    plt.title(title, ha='center', loc='center', size=30)

    # add the correct border around the map - these are known as spines
    for pp in ax.spines.values():
        pp.set_visible(True)
        pp.set_zorder(20)
        pp.set_linewidth(1)
        pp.set_edgecolor("k")
        pp.set_facecolor("k")

    #plt.show()
    #plt.savefig(plotdir + scen + '_' + thresh + '_' + fyears + '_threshold_exceed_consensus_plot.pdf')
    #plt.savefig(plotdir2 + scen + '_' + thresh + '_' + fyears + '_threshold_exceed_consensus_plot.pdf')
    #plt.savefig(plotdir2 + scen + '_' + thresh + '_' + fyears + '_threshold_exceed_consensus_plot.png')
    #plt.savefig(plotdir2 + scen + '_' + thresh + '_' + fyears + '_threshold_exceed_consensus_plot.jpg')
    #plt.close()
    #write plot out here

    plt.savefig(plotdir + scen + '_' + thresh + '_' + fyears + '_threshold_exceed_consensus_plot_mskd.pdf')
    plt.savefig(plotdir2 + scen + '_' + thresh + '_' + fyears + '_threshold_exceed_consensus_plot_mskd.pdf')
    plt.savefig(plotdir2 + scen + '_' + thresh + '_' + fyears + '_threshold_exceed_consensus_plot_mskd.png')
    plt.savefig(plotdir2 + scen + '_' + thresh + '_' + fyears + '_threshold_exceed_consensus_plot_mskd.jpg')
    plt.close()


def add_plot_features(ax, extent=None):
    """ Add some nice cartopy features to a map

    Inputs
    * ax
        current axes
    * extent
        if set, change to the map view to this [x0, x1, y0, y1]
    """
    ax.add_feature(cartopy.feature.OCEAN, facecolor='white', edgecolor='white', zorder=10)
    ax.add_feature(cartopy.feature.BORDERS, facecolor='None', edgecolor='k', lw=0.5, zorder=10)
    ax.coastlines(zorder=11)

    if extent:
        ax.set_extent(extent)



def main():
    
    setup = SETUP1
    dir = "/scratch/dkelley/future_ffdi/"
    indir = dir + "/data/threshold_exceedance/"
    outdir = dir + 'outputs/thresh_exceed_anomaly/'
    plotdir = dir + 'figs/consensus_plots/'
    plotdir2 = dir + 'figs/consensus_plots_new/'
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
    scenario = ['rcp2_6', 'rcp8_5']
    scenario_labels = ['RCP2.6', 'RCP8.5']
    threshold = ['0-11', '4-15', '12-23', '23-10000000']#,'SEVERE']
    threshold_labels = ['Low Mod.', 'Control Burn', 'High', 'Very High']#, 'Severe']
    byears = '1986_2005'
    baseline_label = '1986-2005'
    fyears = ['1_5', '2_deg', '4_deg']
    gwl_labels = ['1.5 degrees', '2.0 degrees', '4.0 degrees']

    #read in mask for non burnable land - Uses the SAGE potential vegetation database and calculates grid boxes where >50% is bare soil and masks out ice,
    #calculated in /net/home/h05/hadin/fcm_durban/DECC_durban/Fire/non_veg_mask.pro;
    mask_dir = '/project/climate_impacts/durban/wp3_results/fire/SAGE_bare_soil_ice_mask_regrid_gt50.pp'
    non_burnable_land_mask = iris.load_cube(mask_dir)
    #bare_soil_ice_mask = non_burnable_land_mask.copy()
    #bare_soil_ice_mask.data = ma.where(non_burnable_land_mask.data == 1, 0, non_burnable_land_mask.data)
    #bare_soil_ice_mask.data.soften_mask()
    #bare_soil_ice_mask.data.mask = ma.nomask
    #bare_soil_ice_mask.data = np.where(bare_soil_ice_mask.data > 1e+20, 1, bare_soil_ice_mask.data)

    bare_soil_ice_mask = non_burnable_land_mask.copy()
    bare_soil_ice_mask.data = ma.where(non_burnable_land_mask.data == 1, np.nan, 1)
    #bare_soil_ice_mask.data = ma.where(non_burnable_land_mask.data == 1, 9.96920997e+36, 1)

    #bare_soil_ice_mask.data.soften_mask()
    bare_soil_ice_mask.data.mask = ma.nomask
    print(bare_soil_ice_mask.data)

    #loop over the thresholds
    for t, thresh in enumerate(threshold):
        print(thresh)

        #loop over the scenarios
        for s, scen in enumerate(scenario):
            #print(s, scen)
            if scen == 'rcp2_6':
                ensemble_name = fut_rcp2_6
                print(ensemble_name)
                fyears = ['1_5', '2_deg']
            elif scen == 'rcp8_5':
                ensemble_name = fut_rcp8_5
                print(ensemble_name)
                fyears = ['1_5', '2_deg', '4_deg']
            else:
                print(s, scen)
                raise Exception('check scenario name')

            #loop over the global warming levels
            for g, gwl in enumerate(fyears):

                title = scenario_labels[s] + ' Change in the number of days above the \n' + threshold_labels[t] + ' FFDI threshold for ' + gwl_labels[g] + ' - (' + baseline_label + ')'

                #check for missing files as not all ensemble members reach each GWL
                #check_files = glob.glob(indir + scen + '/future_threshold_exceedance_' + scen + '_*_' + gwl + '*.nc')
                if g == 0:
                    ensemble = ensemble_name
                    x = len(ensemble_name)
                else:
                    check_files = glob.glob(indir + scen + '/future_threshold_exceedance' + thresh + '_' + scen + '_*_' + gwl + '*.nc')
                    #print(check_files)
                    #print('HELLLOO')
                    check_files.sort()
                    #print(check_files)

                    x = len(check_files) #for adding to the legend of the plots
                    #print(x)
                    if x < 57:
                        new_ensemble = []
                        for string in check_files:
                            print('s')
                            print(s)

                            a = [i for i in ensemble_name if i in string]
                            if len(a) == 1:
                                a = a[0]
                            else:
                                set_trace()
                            new_ensemble.append(a)
                        #print(new_ensemble)
                        ensemble = new_ensemble
                        #raise Exception
                    else:
                        ensemble = ensemble_name

                # 1. plot the consensus map
                plot_single_ensemble_consenus(setup, ensemble, byears, gwl, title, thresh, scen, indir, outdir, plotdir, plotdir2, bare_soil_ice_mask)

    # 2. plot the colorbar
    setup.plot_colorbar(show=False)
    plt.savefig(plotdir + 'master_consensus_plot_colourbar.pdf')
    plt.savefig(plotdir2 + 'master_consensus_plot_colourbar.pdf')
    plt.savefig(plotdir2 + 'master_consensus_plot_colourbar.png')
    plt.savefig(plotdir2 + 'master_consensus_plot_colourbar.jpg')
    #plt.show()


if __name__ == '__main__':
    main()
