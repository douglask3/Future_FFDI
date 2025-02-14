# Reads in burned area from CCI Fire, and plots globally
import os, sys
import iris
import iris.quickplot as qplt
import iris.plot as iplt
import iris.analysis.cartography
import iris.coord_categorisation
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.ticker as mticker
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from iris.experimental.equalise_cubes import equalise_attributes
import numpy as np
import numpy.ma as ma
import datetime as dt

def periodConstraint(cube, t1, t2):
    # Constrains the cube according to min and max datetimes
    def make_time_func(t1m, t2m):
        def tfunc(cell):
            return t1m <= cell.point <= t2m
        return tfunc
    tfunc = make_time_func(t1, t2)
    tconst = iris.Constraint(time=tfunc)
    ocube = cube.extract(tconst)
    return(ocube)

def getData(varname, start=None, end=None):

    path = '/project/LandCoverCCI/V2/Burned/Data/'
    yrfiles = path + '*/*.nc'
    cubes = iris.load(yrfiles, varname)
    equalise_attributes(cubes)
    cube = cubes.concatenate_cube()
    iris.coord_categorisation.add_year(cube, 'time', name='year')
    if start and end:
        ocube = periodConstraint(cube, start, end)
    else:
        ocube = cube

    return ocube

def readData(type='max', start=None, end=None):

    # The fraction of burnable area changes annually with the land cover
    # Meaning there is no month to month variation
    burnable = getData('fraction of burnable area', start=start, end=end)
    observed = getData('fraction of observed area', start=start, end=end)
    burned = getData('burned_area', start=start, end=end)

    burnable_ann = burnable.aggregated_by('year', iris.analysis.SUM)
    burnable_ann_min = burnable_ann.collapsed('time', iris.analysis.MIN)
    burnable_ann_mean = burnable_ann.collapsed('time', iris.analysis.MEAN)
    burnable_ann_max = burnable_ann.collapsed('time', iris.analysis.MAX)
    burnable_tot = burnable.collapsed('time', iris.analysis.SUM) / len(burnable.coord('time').points)

    observed_tot = observed.collapsed('time', iris.analysis.SUM) / len(observed.coord('time').points)

    burned_ann = burned.aggregated_by('year', iris.analysis.SUM)
    burned_ann_min = burned_ann.collapsed('time', iris.analysis.MIN)
    burned_ann_mean = burned_ann.collapsed('time', iris.analysis.MEAN)
    burned_ann_max = burned_ann.collapsed('time', iris.analysis.MAX)

    grid_areas = iris.analysis.cartography.area_weights(burned_ann_mean)
    cube_areas = burned_ann_mean.copy(grid_areas)

    burned_min_perc = 100 * burned_ann_min / cube_areas
    burned_mean_perc = 100 * burned_ann_mean / cube_areas
    burned_max_perc = 100 * burned_ann_max / cube_areas

    if type == 'max':
        outcube = burned_max_perc
    elif type == 'min':
        outcube = burned_min_perc
    elif type == 'mean':
        outcube = burned_mean_perc
    else:
        outcube = burned_max_perc

    # Mask not burnable
    outcube.data = ma.masked_where(burnable_ann_min.data < 1, outcube.data)

    return outcube

def plot(incube, ofile, start=None, end=None, type='max'):

    # Create title
    title_dict = {'min': 'Minimum', 'mean': 'Mean', 'max': 'Maximum'}
    yrtxt = ' ('+str(start.year) + '-' + str(end.year)+')' if start and end else ''
    title = title_dict[type] + ' Annual Burned Area'+yrtxt+'\nas a % of the Total Burnable Area'
    # Now do the plotting
    fig = plt.figure(figsize=(12,7), dpi=96)

    bounds = np.arange(0, 120, 10)
    # norm = mpl.colors.BoundaryNorm(bounds, ncolors=len(bounds))
    # norm = colors.LogNorm(vmin=0, vmax=bounds[-1])
    # pcm = iplt.pcolormesh(incube, norm=norm, cmap='inferno')
    pcm = iplt.pcolormesh(incube, cmap='inferno_r', vmin=0, vmax=bounds[-1])
    plt.title(title, fontsize=20)
    plt.xlabel('longitude / degrees')
    plt.ylabel('latitude / degrees')
    var_plt_ax = plt.gca()

    var_plt_ax.set_extent([-180,180,-90,90])
    borderlines = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_0_boundary_lines_land',
        scale='50m',
        facecolor='none')
    var_plt_ax.add_feature(borderlines, edgecolor='black', alpha=0.5)
    var_plt_ax.coastlines(resolution='110m', color='black')
    # gl = var_plt_ax.gridlines(color="gray", alpha=0.2, draw_labels=True)
    # gl.xlabels_top = False
    # gl.ylabels_left = False
    # gl.xformatter = LONGITUDE_FORMATTER
    # gl.yformatter = LATITUDE_FORMATTER
    # gl.ylocator = mticker.FixedLocator(np.arange(-90,120, 30))
    # gl.xlabel_style = {'size': 8}
    # gl.ylabel_style = {'size': 8}

    vleft, vbottom, vwidth, vheight = var_plt_ax.get_position().bounds
    plt.gcf().subplots_adjust(top=vbottom + vheight, bottom=vbottom + 0.04,
                              left=vleft, right=vleft + vwidth)
    cbar_axes = fig.add_axes([((1/3.)*vwidth*(1/2.)) + vleft, vbottom - 0.025, (2/3.)*vwidth, 0.025])
    # cbar = plt.colorbar(pcm, norm=norm, boundaries=bounds, cax=cbar_axes, orientation='horizontal', extend='max')
    cbar = plt.colorbar(pcm, boundaries=bounds, cax=cbar_axes, orientation='horizontal', extend='max')
    cbar.set_label('% Burned', size=16)
    cbar.ax.tick_params(length=0)
    cbar.ax.tick_params(labelsize=14)

    fig.savefig(ofile, bbox_inches='tight')
    plt.close(fig)


def main():
    # Read data and plot
    periods = [(dt.datetime(2001, 1, 1), dt.datetime(2018, 12, 31))]  # (dt.datetime(2001, 1, 1), dt.datetime(2009, 12, 31))
    for start, end in periods:
        for stat in ['mean', 'max']:
            cube = readData(type=stat, start=start, end=end)
            ofile = 'plots/annual_burned_'+str(start.year)+'-'+str(end.year)+'_'+stat+'.png'
            plot(cube, ofile, start=start, end=end, type=stat)


if __name__ == '__main__':
    main()

