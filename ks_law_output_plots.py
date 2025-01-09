import numpy as np
import utilities_statistics as us 
import common
import h5py

dir_data = 'Runs/'
#model_name = 'L0100N0752/Thermal_non_equilibrium'
model_name = 'L0050N0752/Thermal_non_equilibrium'

model_name_highres =  'L0012N0376/Thermal_non_equilibrium'
model_name_lowhres =  'L0050N0752/Thermal_non_equilibrium'
model_name_method =  'L0025N0376/Thermal_non_equilibrium'

plot_all_methods = True

#choose the type of profile to be read
#method = 'spherical_apertures'
method = 'circular_apertures_face_on_map'
#method = 'circular_apertures_random_map'
dr = 1.0

outdir = dir_data + model_name + '/Plots/'
plt = common.load_matplotlib()

def plot_Hneutral_obs(ax):
    xHI, SFR, SFRdn, SFRup =  np.loadtxt('data/SFLawHneutral_NobelsComp.txt', unpack = True, usecols = [0,1,2,3])
    ax.errorbar(xHI[0:4], SFR[0:4], yerr=[SFR[0:4] - SFRdn[0:4], SFRup[0:4] - SFR[0:4]], marker='D', color='navy', ls='None', fillstyle='none', markersize=5, label='Bigiel+08')
    ax.errorbar(xHI[4:len(xHI)], SFR[4:len(xHI)], yerr=[SFR[4:len(xHI)] - SFRdn[4:len(xHI)], SFRup[4:len(xHI)] - SFR[4:len(xHI)]], marker='^', color='MediumBlue', ls='None', fillstyle='none', markersize=5, label='Bigiel+10')


def plot_HI_obs(ax):
    xHI, SFR, SFRdn, SFRup =  np.loadtxt('data/Wang24_SFLHI.csv', unpack = True, usecols = [0,1,2,3])
    ax.errorbar(xHI, SFR, yerr=[SFR - SFRdn, SFRup - SFR], marker='o', color='DarkGreen', ls='None', fillstyle='none', markersize=5, label='Wang+24')

    xHI, SFR, SFRdn, SFRup =  np.loadtxt('data/Walter08_SFLH1.csv', unpack = True, usecols = [0,1,2,3])
    ax.errorbar(xHI, SFR, yerr=[SFR - SFRdn, SFRup - SFR], marker='D', color='Teal', ls='None', fillstyle='none', markersize=5, label='Walter+08')

def plot_H2_obs(ax):
    xH2, SFR, SFRdn, SFRup =  np.loadtxt('data/Leroy13_SFLH2.csv', unpack = True, usecols = [0,1,2,3])
    ax.errorbar(xH2 - np.log10(1.36), SFR, yerr=[SFR - SFRdn, SFRup - SFR], marker='s', color='Red', ls='None', fillstyle='none', markersize=5, label='Leroy+13 var $\\alpha_{\\rm CO}$')
    xH2, SFR, SFRdn, SFRup =  np.loadtxt('data/Leroy13_SFLH2_fixedAlphaCO.csv', unpack = True, usecols = [0,1,2,3])
    ax.errorbar(xH2 - np.log10(1.36), SFR, yerr=[SFR - SFRdn, SFRup - SFR], marker='*', color='Red', ls='None', fillstyle='none', markersize=5, label='Leroy+13 fix $\\alpha_{\\rm CO}$')
    #xE20 = [0.50104,2.70028] 
    #yE20 = [-2.60337,-0.315380]
    #ax.plot(xE20, yE20, linestyle='dashed',color='CornflowerBlue', label='Ellison+20', lw = 4)
    xE20, yE20 = np.loadtxt('data/Ellison20_SFLawH2.txt', unpack = True, usecols=[0,1])
    xE20 = xE20 - 6
    #ax.plot(xE20[0:25], yE20[0:25], linestyle='solid',color='CornflowerBlue', lw = 1.5, label='Ellison+20')
    #ax.plot(xE20[25:53], yE20[25:53], linestyle='solid',color='CornflowerBlue', lw = 1.5)
    #ax.plot(xE20[53:89], yE20[53:89], linestyle='solid',color='CornflowerBlue', lw = 1.5)


    f = h5py.File('data/SpatiallyResolvedMolecularKSRelation/Querejeta2021.hdf5', 'r')
    xplot = f['x/values']
    xerr = f['x/scatter']
    xerrdn = np.log10(xplot[:]) - np.log10(xplot[:] - xerr[0,:])
    xerrup = np.log10(xplot[:] + xerr[1,:]) - np.log10(xplot[:])
   
    yplot = f['y/values']
    yerr = f['y/scatter']
    yerrdn = np.log10(yplot[:]) - np.log10(yplot[:] - yerr[0,:])
    yerrup = np.log10(yplot[:] + yerr[1,:]) - np.log10(yplot[:])
    ax.errorbar(np.log10(xplot[:]), np.log10(yplot[:]), xerr=[xerrdn, xerrup], yerr=[yerrdn, yerrup], marker='P', color='CadetBlue', ls='None', fillstyle='none', markersize=5, label='Querejeta+21')

    f = h5py.File('data/SpatiallyResolvedMolecularKSRelation/Ellison2020.hdf5', 'r')
    xplot = f['x/values']
    xerr = f['x/scatter']
    xerrdn = np.log10(xplot[:]) - np.log10(xplot[:] - xerr[0,:])
    xerrup = np.log10(xplot[:] + xerr[1,:]) - np.log10(xplot[:])
   
    yplot = f['y/values']
    yerr = f['y/scatter']
    yerrdn = np.log10(yplot[:]) - np.log10(yplot[:] - yerr[0,:])
    yerrup = np.log10(yplot[:] + yerr[1,:]) - np.log10(yplot[:])
    yplot = yplot[:]
    xplot = xplot[:]
    ind =np.where(np.isnan(yplot) == False)
    ax.errorbar(np.log10(xplot[ind]), np.log10(yplot[ind]), xerr=[xerrdn[ind], xerrup[ind]], yerr=[yerrdn[ind], yerrup[ind]], marker='P', color='CornflowerBlue', ls='None', fillstyle='none', markersize=5, label='Ellison+20')



def plot_lines_constant_deptime(ax, xmin, xmax):

    times = [1e8, 1e9, 1e10]
    labels = ['0.1Gyr', '1Gyr', '10Gyr']
    for j in range(0,len(times)):
       ymin = np.log10(10**xmin * 1e6 / times[j])
       ymax = np.log10(10**xmax * 1e6 / times[j])
       ax.text(1.9, np.log10(10**2.1 * 1e6 / times[j]), labels[j])
       ax.plot([xmin, xmax], [ymin,ymax], linestyle='dotted', color='k')

def plot_gas_resolutions(ax, gas_type='HI', label = True):

    data_circ_rand = np.loadtxt(dir_data + model_name + '/ProcessedData/' + gas_type + 'SFLaw_z0.0_' + method + '.txt')
    data_circ_face = np.loadtxt(dir_data + model_name_highres + '/ProcessedData/' + gas_type + 'SFLaw_z0.0_' + method + '.txt')
    data_sph = np.loadtxt(dir_data + model_name_lowhres + '/ProcessedData/' + gas_type + 'SFLaw_z0.0_'+ method + '.txt')

    ax.plot(data_circ_rand[:,0], data_circ_rand[:,1], linestyle = 'solid', color='green', label = 'm7' if label else None)
    ax.fill_between(data_circ_rand[:,0], data_circ_rand[:,1] - data_circ_rand[:,2], data_circ_rand[:,1] + data_circ_rand[:,3], facecolor = 'green', alpha=0.2, interpolate=True) 
    ax.plot(data_circ_face[:,0], data_circ_face[:,1], linestyle = 'solid', color='red', label = 'm5' if label else None)
    ax.fill_between(data_circ_face[:,0], data_circ_face[:,1] - data_circ_face[:,2], data_circ_face[:,1] + data_circ_face[:,3], facecolor = 'red', alpha=0.2, interpolate=True)
    ax.plot(data_sph[:,0], data_sph[:,1], linestyle = 'solid', color='blue', label = 'm6' if label else None)
    ax.fill_between(data_sph[:,0], data_sph[:,1] - data_sph[:,2], data_sph[:,1] + data_sph[:,3], facecolor = 'blue', alpha=0.2, interpolate=True)


def plot_gas_all_methods(ax, gas_type='HI', label = True):

    data_circ_rand = np.loadtxt(dir_data + model_name_method + '/ProcessedData/' + gas_type + 'SFLaw_z0.0_circular_apertures_random_map.txt')
    data_circ_face = np.loadtxt(dir_data + model_name_method + '/ProcessedData/' + gas_type + 'SFLaw_z0.0_circular_apertures_face_on_map.txt')
    data_sph = np.loadtxt(dir_data + model_name_method + '/ProcessedData/' + gas_type + 'SFLaw_z0.0_spherical_apertures.txt')
    data_grid = np.loadtxt(dir_data + model_name_method + '/ProcessedData/' + gas_type + 'SFLaw_z0.0_grid_random_map.txt')

    ax.plot(data_circ_rand[:,0], data_circ_rand[:,1], linestyle = 'solid', color='green', label = 'annuli-rand' if label else None)
    ax.fill_between(data_circ_rand[:,0], data_circ_rand[:,1] - data_circ_rand[:,2], data_circ_rand[:,1] + data_circ_rand[:,3], facecolor = 'green', alpha=0.2, interpolate=True) 
    ax.plot(data_circ_face[:,0], data_circ_face[:,1], linestyle = 'solid', color='red', label = 'annuli-face' if label else None)
    ax.fill_between(data_circ_face[:,0], data_circ_face[:,1] - data_circ_face[:,2], data_circ_face[:,1] + data_circ_face[:,3], facecolor = 'red', alpha=0.2, interpolate=True)
    #ax.plot(data_sph[:,0], data_sph[:,1], linestyle = 'solid', color='blue', label = 'Spherical' if label else None)
    #ax.fill_between(data_sph[:,0], data_sph[:,1] - data_sph[:,2], data_sph[:,1] + data_sph[:,3], facecolor = 'blue', alpha=0.2, interpolate=True)
    ax.plot(data_grid[:,0], data_grid[:,1], linestyle = 'solid', color='blue', label = 'grid-rand' if label else None)
    ax.fill_between(data_grid[:,0], data_grid[:,1] - data_grid[:,2], data_grid[:,1] + data_grid[:,3], facecolor = 'blue', alpha=0.2, interpolate=True)

def plot_gas_ssfr(ax, gas_type='HI', label = True):

    data_circ_rand = np.loadtxt(dir_data + model_name + '/ProcessedData/' + gas_type + 'SFLaw_z0.0_' + method + '_ssfr_low.txt')
    data_circ_face = np.loadtxt(dir_data + model_name + '/ProcessedData/' + gas_type + 'SFLaw_z0.0_' + method + '_ssfr_inter.txt')
    data_sph = np.loadtxt(dir_data + model_name + '/ProcessedData/' + gas_type + 'SFLaw_z0.0_' + method + '_ssfr_high.txt')

    ax.plot(data_circ_rand[:,0], data_circ_rand[:,1], linestyle = 'solid', color='red', label = 'bottom 33rd sSFR' if label else None)
    ax.fill_between(data_circ_rand[:,0], data_circ_rand[:,1] - data_circ_rand[:,2], data_circ_rand[:,1] + data_circ_rand[:,3], facecolor = 'red', alpha=0.2, interpolate=True) 
    ax.plot(data_circ_face[:,0], data_circ_face[:,1], linestyle = 'solid', color='green', label = '33rd-66th sSFR' if label else None)
    ax.fill_between(data_circ_face[:,0], data_circ_face[:,1] - data_circ_face[:,2], data_circ_face[:,1] + data_circ_face[:,3], facecolor = 'green', alpha=0.2, interpolate=True)
    ax.plot(data_sph[:,0], data_sph[:,1], linestyle = 'solid', color='blue', label = 'above 66th sSFR' if label else None)
    ax.fill_between(data_sph[:,0], data_sph[:,1] - data_sph[:,2], data_sph[:,1] + data_sph[:,3], facecolor = 'blue', alpha=0.2, interpolate=True)

def plot_gas_galtype(ax, gas_type='HI', label = True):

    data_circ_rand = np.loadtxt(dir_data + model_name + '/ProcessedData/' + gas_type + 'SFLaw_z0.0_' + method + '_satellites.txt')
    data_circ_face = np.loadtxt(dir_data + model_name + '/ProcessedData/' + gas_type + 'SFLaw_z0.0_' + method + '_centrals.txt')

    ax.plot(data_circ_rand[:,0], data_circ_rand[:,1], linestyle = 'solid', color='red', label = 'satellites' if label else None)
    ax.fill_between(data_circ_rand[:,0], data_circ_rand[:,1] - data_circ_rand[:,2], data_circ_rand[:,1] + data_circ_rand[:,3], facecolor = 'red', alpha=0.2, interpolate=True) 
    ax.plot(data_circ_face[:,0], data_circ_face[:,1], linestyle = 'solid', color='blue', label = 'centrals' if label else None)
    ax.fill_between(data_circ_face[:,0], data_circ_face[:,1] - data_circ_face[:,2], data_circ_face[:,1] + data_circ_face[:,3], facecolor = 'blue', alpha=0.2, interpolate=True)


def plot_gas_all_redshifts(ax, gas_type='HI', label = True):

    #redshifts = ['0.0', '1.0', '2.0', '3.5', '4.0', '5.0']
    #cols = ['DarkRed', 'Salmon', 'Olive', 'YellowGreen', 'MediumBlue', 'Indigo']
    #redshifts = ['0.0', '1.0', '3.0', '4.0', '5.0']
    #cols = ['DarkRed', 'Salmon', 'YellowGreen', 'MediumBlue', 'Indigo']
    redshifts = ['0.0', '1.0', '2.0', '3.0']
    cols = ['DarkRed', 'Olive', 'MediumBlue', 'Indigo']

    for i,z in enumerate(redshifts):
        data = np.loadtxt(dir_data + model_name + '/ProcessedData/' + gas_type + 'SFLaw_z' + z + '_' + method + '.txt')
        ax.plot(data[:,0], data[:,1], linestyle = 'solid', color=cols[i], label = 'z='+ z if label else None)
        ax.fill_between(data[:,0], data[:,1] - data[:,2], data[:,1] + data[:,3], facecolor = cols[i], alpha=0.2, interpolate=True) 

    return cols
####################### plot all methods at z=0 #########################################
min_gas_dens = -2
fig = plt.figure(figsize=(14,6))
xtits = ["$\\rm log_{10} (\\rm \\Sigma_{\\rm HI}/M_{\\odot}\\, pc^{-2})$", "$\\rm log_{10} (\\rm \\Sigma_{\\rm H_{2}}/M_{\\odot}\\, pc^{-2})$", "$\\rm log_{10} (\\rm \\Sigma_{\\rm HI+H_{2}}/M_{\\odot}\\, pc^{-2})$"]
ytits = ["$\\rm log_{10} (\\rm \\Sigma_{\\rm SFR}/M_{\\odot}\\, yr^{-1}\\, kpc^{-2})$", "",""]
xmin, xmax, ymin, ymax = min_gas_dens, 2.5, -6, 0.5
subplots = [131, 132, 133]


for i,s in enumerate(subplots):
    ax = fig.add_subplot(s)
    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtits[i], ytits[i], locators=(0.5, 0.5, 0.5, 0.5))

    if i == 0:
        plot_gas_resolutions(ax, gas_type='HI')
        plot_HI_obs(ax)
        plot_lines_constant_deptime(ax, xmin, xmax)
        common.prepare_legend(ax, ['green','red','blue','DarkGreen','Teal'], loc = 2)
    if i == 1:
        plot_gas_resolutions(ax, gas_type='H2', label = False)
        plot_H2_obs(ax)
        plot_lines_constant_deptime(ax, xmin, xmax)
        common.prepare_legend(ax, ['Red', 'Red', 'CadetBlue','CornflowerBlue'], loc = 2)

    if i == 2:
        plot_gas_resolutions(ax, gas_type='Hneutral', label = False)
        plot_Hneutral_obs(ax)
        plot_lines_constant_deptime(ax, xmin, xmax)
        common.prepare_legend(ax, ['navy','MediumBlue'], loc = 2)

common.savefig(outdir, fig, 'AllResolutions_KSLaw_z0.pdf')


####################### plot all methods at z=0 #########################################
min_gas_dens = -2

if(plot_all_methods):
   fig = plt.figure(figsize=(14,6))
   xtits = ["$\\rm log_{10} (\\rm \\Sigma_{\\rm HI}/M_{\\odot}\\, pc^{-2})$", "$\\rm log_{10} (\\rm \\Sigma_{\\rm H_{2}}/M_{\\odot}\\, pc^{-2})$", "$\\rm log_{10} (\\rm \\Sigma_{\\rm HI+H_{2}}/M_{\\odot}\\, pc^{-2})$"]
   ytits = ["$\\rm log_{10} (\\rm \\Sigma_{\\rm SFR}/M_{\\odot}\\, yr^{-1}\\, kpc^{-2})$", "",""]
   xmin, xmax, ymin, ymax = min_gas_dens, 2.5, -6, 0.5
   subplots = [131, 132, 133]
  
  
   for i,s in enumerate(subplots):
       ax = fig.add_subplot(s)
       common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtits[i], ytits[i], locators=(0.5, 0.5, 0.5, 0.5))
  
       if i == 0:
           plot_gas_all_methods(ax, gas_type='HI')
           plot_HI_obs(ax)
           plot_lines_constant_deptime(ax, xmin, xmax)
           common.prepare_legend(ax, ['green','red','blue','DarkGreen','Teal'], loc = 2)
       if i == 1:
           plot_gas_all_methods(ax, gas_type='H2', label = False)
           plot_H2_obs(ax)
           plot_lines_constant_deptime(ax, xmin, xmax)
           common.prepare_legend(ax, ['Red', 'Red', 'CadetBlue','CornflowerBlue'], loc = 2)
  
       if i == 2:
           plot_gas_all_methods(ax, gas_type='Hneutral', label = False)
           plot_Hneutral_obs(ax)
           plot_lines_constant_deptime(ax, xmin, xmax)
           common.prepare_legend(ax, ['navy','MediumBlue'], loc = 2)
  
   common.savefig(outdir, fig, 'AllMethods_KSLaw_z0.pdf')

####################### plot different sSFR files at z=0 #########################################
min_gas_dens = -2
fig = plt.figure(figsize=(14,6))
xtits = ["$\\rm log_{10} (\\rm \\Sigma_{\\rm HI}/M_{\\odot}\\, pc^{-2})$", "$\\rm log_{10} (\\rm \\Sigma_{\\rm H_{2}}/M_{\\odot}\\, pc^{-2})$", "$\\rm log_{10} (\\rm \\Sigma_{\\rm HI+H_{2}}/M_{\\odot}\\, pc^{-2})$"]
ytits = ["$\\rm log_{10} (\\rm \\Sigma_{\\rm SFR}/M_{\\odot}\\, yr^{-1}\\, kpc^{-2})$", "",""]
xmin, xmax, ymin, ymax = min_gas_dens, 2.5, -6, 0.5
subplots = [131, 132, 133]


for i,s in enumerate(subplots):
    ax = fig.add_subplot(s)
    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtits[i], ytits[i], locators=(0.5, 0.5, 0.5, 0.5))

    if i == 0:
        plot_gas_ssfr(ax, gas_type='HI', label = True)
        plot_HI_obs(ax)
        plot_lines_constant_deptime(ax, xmin, xmax)
        common.prepare_legend(ax, ['red','green','blue','DarkGreen','Teal'], loc = 2)
    if i == 1:
        plot_gas_ssfr(ax, gas_type='H2', label = False)
        plot_H2_obs(ax)
        plot_lines_constant_deptime(ax, xmin, xmax)
        common.prepare_legend(ax, ['Red', 'Red', 'CadetBlue','CornflowerBlue'], loc = 2)

    if i == 2:
        plot_gas_ssfr(ax, gas_type='Hneutral', label = False)
        plot_Hneutral_obs(ax)
        plot_lines_constant_deptime(ax, xmin, xmax)
        common.prepare_legend(ax, ['navy','MediumBlue'], loc = 2)

common.savefig(outdir, fig, 'GlobalSSFR_KSLaw_z0_' + method + '.pdf')

####################### plot satellites/centrals separately at z=0 #########################################
min_gas_dens = -2
fig = plt.figure(figsize=(14,6))
xtits = ["$\\rm log_{10} (\\rm \\Sigma_{\\rm HI}/M_{\\odot}\\, pc^{-2})$", "$\\rm log_{10} (\\rm \\Sigma_{\\rm H_{2}}/M_{\\odot}\\, pc^{-2})$", "$\\rm log_{10} (\\rm \\Sigma_{\\rm HI+H_{2}}/M_{\\odot}\\, pc^{-2})$"]
ytits = ["$\\rm log_{10} (\\rm \\Sigma_{\\rm SFR}/M_{\\odot}\\, yr^{-1}\\, kpc^{-2})$", "",""]
xmin, xmax, ymin, ymax = min_gas_dens, 2.5, -6, 0.5
subplots = [131, 132, 133]


for i,s in enumerate(subplots):
    ax = fig.add_subplot(s)
    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtits[i], ytits[i], locators=(0.5, 0.5, 0.5, 0.5))

    if i == 0:
        plot_gas_galtype(ax, gas_type='HI', label = True)
        plot_HI_obs(ax)
        plot_lines_constant_deptime(ax, xmin, xmax)
        common.prepare_legend(ax, ['red','blue','DarkGreen','Teal'], loc = 2)
    if i == 1:
        plot_gas_galtype(ax, gas_type='H2', label = False)
        plot_H2_obs(ax)
        plot_lines_constant_deptime(ax, xmin, xmax)
        common.prepare_legend(ax, ['Red', 'Red', 'CadetBlue','CornflowerBlue'], loc = 2)

    if i == 2:
        plot_gas_galtype(ax, gas_type='Hneutral', label = False)
        plot_Hneutral_obs(ax)
        plot_lines_constant_deptime(ax, xmin, xmax)
        common.prepare_legend(ax, ['navy','MediumBlue'], loc = 2)

common.savefig(outdir, fig, 'GalaxyType_KSLaw_z0_' + method + '.pdf')


####################### plot all redshifts for one method #########################################
min_gas_dens = -2
fig = plt.figure(figsize=(14,6))
xtits = ["$\\rm log_{10} (\\rm \\Sigma_{\\rm HI}/M_{\\odot}\\, pc^{-2})$", "$\\rm log_{10} (\\rm \\Sigma_{\\rm H_{2}}/M_{\\odot}\\, pc^{-2})$", "$\\rm log_{10} (\\rm \\Sigma_{\\rm HI+H_{2}}/M_{\\odot}\\, pc^{-2})$"]
ytits = ["$\\rm log_{10} (\\rm \\Sigma_{\\rm SFR}/M_{\\odot}\\, yr^{-1}\\, kpc^{-2})$", "",""]
xmin, xmax, ymin, ymax = min_gas_dens, 2.5, -6, 1.5
subplots = [131, 132, 133]


for i,s in enumerate(subplots):
    ax = fig.add_subplot(s)
    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtits[i], ytits[i], locators=(0.5, 0.5, 0.5, 0.5))

    if i == 0:
        cols = plot_gas_all_redshifts(ax, gas_type='HI', label = True)
        plot_HI_obs(ax)
        plot_lines_constant_deptime(ax, xmin, xmax)
        #common.prepare_legend(ax, ['DarkRed', 'Salmon', 'Olive', 'YellowGreen', 'MediumBlue', 'Indigo','DarkGreen','Teal'], loc = 2)
        common.prepare_legend(ax, cols, loc = 2)

    if i == 1:
        _ = plot_gas_all_redshifts(ax, gas_type='H2', label = False)
        plot_H2_obs(ax)
        plot_lines_constant_deptime(ax, xmin, xmax)
        common.prepare_legend(ax, ['Red', 'Red', 'CadetBlue','CornflowerBlue'], loc = 2)

    if i == 2:
        _ = plot_gas_all_redshifts(ax, gas_type='Hneutral', label = False)
        plot_Hneutral_obs(ax)
        plot_lines_constant_deptime(ax, xmin, xmax)
        common.prepare_legend(ax, ['navy','MediumBlue'], loc = 2)

common.savefig(outdir, fig, 'Evolution_KSLaw_' + method + '.pdf')


