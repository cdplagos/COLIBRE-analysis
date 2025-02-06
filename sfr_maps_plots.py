import numpy as np
import utilities_statistics as us
import common
import h5py
from hyperfit.linfit import LinFit
from hyperfit.data import ExampleData

#define radial bins of interest. This going from 0 to 50kpc, in bins of 1kpc
dir_data = 'Runs/'
#model_name = 'L0100N0752/Thermal_non_equilibrium'
model_name = 'L0025N0376/Thermal_non_equilibrium'
#model_name = 'L0050N0752/Thermal_non_equilibrium'

#choose the type of profile to be read
#method = 'spherical_apertures'
method = 'circular_apertures_face_on_map'
#method = 'circular_apertures_random_map'
##method = 'grid_random_map'
dr = 1.0
ztarget = '0.0'

minct = 3 #minimum number of datapoints per bin to compute median
n_thresh_bin = 10 #minimum number of particles per annuli or bin to consider datapoint in plots

outdir = dir_data + model_name + '/Plots/'

data = np.loadtxt(dir_data + model_name + '/ProcessedData/' + 'GalaxyProperties_z' + str(ztarget) + '.txt')
idgal        = data[:,0]
typeg        = data[:,1] 
xg           = data[:,2]
yg           = data[:,3]
zg           = data[:,4]
m30          = data[:,5]  
sfr30        = data[:,6]
r50          = data[:,7]
mHI          = data[:,8]  
mH2          = data[:,9]
kappacostar  = data[:,10]
kappacogas   = data[:,11]
disctotot    = data[:,12]  
spin         = data[:,13]
stellarage   = data[:,14]
ZgasLow      = data[:,15]
ZgasHigh     = data[:,16]
mdust        = data[:,17]
ngals = len(idgal)
H2frac_galaxy = mH2*1e10/m30

ssfr_thresh = np.percentile(sfr30/m30, [33,66])

#sfr and stellar mass profiles
sfr_prof = np.loadtxt(dir_data + model_name + '/ProcessedData/' + 'SFR_profiles_ap50ckpc_' + method + '_dr'+ str(dr) + '_z' + ztarget + '.txt')
mstar_prof = np.loadtxt(dir_data + model_name + '/ProcessedData/' + 'Mstar_profiles_ap50ckpc_' + method + '_dr'+ str(dr) + '_z' + ztarget + '.txt')
#gas profiles
mHI_prof = np.loadtxt(dir_data + model_name + '/ProcessedData/' + 'MHI_profiles_ap50ckpc_' + method + '_dr'+ str(dr) + '_z' + ztarget + '.txt')
mH2_prof = np.loadtxt(dir_data + model_name + '/ProcessedData/' + 'MH2_profiles_ap50ckpc_' + method + '_dr'+ str(dr) + '_z' + ztarget + '.txt')
mdust_prof = np.loadtxt(dir_data + model_name + '/ProcessedData/' + 'Mdust_profiles_ap50ckpc_' + method + '_dr'+ str(dr) + '_z' + ztarget + '.txt')
#metallicity profiles
oh_prof = np.loadtxt(dir_data + model_name + '/ProcessedData/' + 'OH_gas_profiles_ap50ckpc_' + method + '_dr'+ str(dr) + '_z' + ztarget + '.txt')
feh_prof = np.loadtxt(dir_data + model_name + '/ProcessedData/' + 'FeH_gas_profiles_ap50ckpc_' + method + '_dr'+ str(dr) + '_z' + ztarget + '.txt')
#number of particles profiles
n0_prof = np.loadtxt(dir_data + model_name + '/ProcessedData/' + 'NumberPart0_profiles_ap50ckpc_' + method + '_dr'+ str(dr) + '_z' + ztarget + '.txt')
n4_prof = np.loadtxt(dir_data + model_name + '/ProcessedData/' + 'NumberPart4_profiles_ap50ckpc_' + method + '_dr'+ str(dr) + '_z' + ztarget + '.txt')
#radial info
rad_prof = np.loadtxt(dir_data + model_name + '/ProcessedData/' + 'radii_info_' + method + '_dr'+ str(dr) + '_z' + ztarget + '.txt')



if((method == 'spherical_apertures') | (method == 'circular_apertures_face_on_map') | (method == 'circular_apertures_random_map')):
    nr = len(rad_prof)
    dr = rad_prof[1] - rad_prof[0]
    area_annuli = (np.pi * (rad_prof + dr/2)**2 - np.pi * (rad_prof - dr/2)**2) #in kpc^2
    for g in range(0,ngals):
        sfr_prof[g,:] = sfr_prof[g,:] / area_annuli[:]
        mstar_prof[g,:] = mstar_prof[g,:] / area_annuli[:]
        mHI_prof[g,:] = mHI_prof[g,:] / area_annuli[:]
        mH2_prof[g,:] = mH2_prof[g,:] / area_annuli[:]
        mdust_prof[g,:] = mdust_prof[g,:] / area_annuli[:]

elif (method == 'grid_random_map'):
    nr = len(rad_prof)
    area_pixel = dr**2
    for g in range(0,ngals):
        sfr_prof[g,:] = sfr_prof[g,:] / area_pixel
        mstar_prof[g,:] = mstar_prof[g,:] / area_pixel
        mHI_prof[g,:] = mHI_prof[g,:] / area_pixel
        mH2_prof[g,:] = mH2_prof[g,:] / area_pixel
        mdust_prof[g,:] = mdust_prof[g,:] / area_pixel

ind = np.where(mstar_prof == 0)
mstar_prof[ind] = 1e-10

#first plot calculate individual arrays with each value for all galaxies to plot KS law for all galaxies using HI, H2 and total (HI+H2)
sfr_prof_total = np.zeros(shape = (ngals * nr))
mst_prof_total = np.zeros(shape = (ngals * nr))
mHI_prof_total = np.zeros(shape = (ngals * nr))
mH2_prof_total = np.zeros(shape = (ngals * nr))
mdust_prof_total = np.zeros(shape = (ngals * nr))
oh_prof_total  = np.zeros(shape = (ngals * nr))
feh_prof_total = np.zeros(shape = (ngals * nr))
n0_prof_total  = np.zeros(shape = (ngals * nr))
n4_prof_total  = np.zeros(shape = (ngals * nr))
gal_prop_mstar = np.zeros(shape = (ngals * nr))
gal_prop_disctotot = np.zeros(shape = (ngals * nr))
gal_prop_kappacostar = np.zeros(shape = (ngals * nr))
gal_prop_kappacogas  = np.zeros(shape = (ngals * nr))
gal_prop_r50 = np.zeros(shape = (ngals * nr))
gal_prop_zgas = np.zeros(shape = (ngals * nr))
gal_prop_ssfr = np.zeros(shape = (ngals * nr))
gal_prop_type = np.zeros(shape = (ngals * nr))
p = 0
for g in range(0,ngals):
    for r in range(0,nr):
        sfr_prof_total[p] = sfr_prof[g,r] 
        mst_prof_total[p] = mstar_prof[g,r] 
        mHI_prof_total[p] = mHI_prof[g,r] 
        mH2_prof_total[p] = mH2_prof[g,r] 
        mdust_prof_total[p] = mdust_prof[g,r]
        oh_prof_total[p]  = oh_prof[g,r]
        feh_prof_total[p] = feh_prof[g,r]
        n0_prof_total[p]  = n0_prof[g,r]
        n4_prof_total[p]  = n4_prof[g,r]
        gal_prop_mstar[p] = m30[g]
        gal_prop_disctotot[p] = disctotot[g]
        gal_prop_kappacostar[p] = kappacostar[g]
        gal_prop_kappacogas[p] = kappacogas[g]
        gal_prop_r50[p] = r50[g] * 1e3 #in ckpc
        gal_prop_zgas[p] = ZgasHigh[g]
        gal_prop_ssfr[p] = sfr30[g] / m30[g]
        gal_prop_type[p] = typeg[g]
        p = p + 1

def compute_3d_relation(prop1, prop2, prop3, limp1, limp2, limp3):

    ## apply limits
    ind = np.where((prop1 >= limp1[0]) & (prop1 <= limp1[1]) & (prop2 >= limp2[0]) & (prop2 <= limp2[1]) & (prop3 >= limp3[0]) & (prop3 <= limp3[1]))
    prop1 = prop1[ind]
    prop2 = prop2[ind]
    prop3 = prop3[ind]

    data = ExampleData()
    data_fit = np.zeros(shape = (3,len(prop1)))
    data_err = np.zeros(shape = (3,len(prop1)))
    data_fit[0,:] = np.log10(prop1)
    data_fit[1,:] = np.log10(prop2)
    data_fit[2,:] = np.log10(prop3)

    data_err[0,:] = np.log10(prop1) - np.log10(0.95 * prop1) #apply a small 5% error
    data_err[1,:] = np.log10(prop2) - np.log10(0.95 * prop2) #apply a small 5% error
    data_err[2,:] = np.log10(prop3) - np.log10(0.95 * prop3) #apply a small 5% error

    data.xs = data_fit
    data.cov = data_err

    hf = LinFit(data.xs, data.cov, weights=data.weights)
    bounds = ((-5.0, 5.0), (-10.0, 10.0), (-5.0, 5.0), (1.0e-5, 5.0))
    mcmc_samples, mcmc_lnlike = hf.emcee(bounds, verbose=False)
    sigmas = hf.get_sigmas()

    print("best fit:", np.mean(mcmc_samples, axis=1), np.std(mcmc_samples, axis=1))


#compute_3d_relation(mH2_prof_total*1e-6, mst_prof_total*1e-6, sfr_prof_total, [1, 1e7], [0.1, 1e4], [1e-6, 1e2])


def compute_median_relations(x, y, nbins, add_last_bin):
    result, x = us.wmedians_variable_bins(x=x, y=y, nbins=nbins, add_last_bin = add_last_bin)
    return result, x

def plot_KS_relation_nogradients(ax, n0_prof_total, sigma_sfr, sigma_gas, min_gas_dens = -2):

    ind = np.where((sigma_sfr != 0) & (sigma_gas != 0) & (n0_prof_total >= n_thresh_bin))
    y, x = compute_median_relations(np.log10(sigma_gas[ind] * 1e-6), np.log10(sigma_sfr[ind]), nbins = 20, add_last_bin = True)
 
    ind = np.where(y[0,:] != 0)
    yplot = y[0,ind]
    errdn = y[1,ind]
    errup = y[2,ind]
    ax.fill_between(x[ind],errdn[0], errup[0], color='k', alpha=0.2)
    ax.plot(x[ind],yplot[0], linestyle='solid', color='k')

def plot_KS_relation(ax, n0_prof_total, sigma_sfr, sigma_gas, third_prop, vmin = -1, vmax=2, density = True, min_gas_dens = -2, save_to_file = False, file_name = 'SFlaw.txt'):

    ind = np.where((sigma_sfr != 0) & (sigma_gas != 0) & (n0_prof_total >= n_thresh_bin))
    y, x = compute_median_relations(np.log10(sigma_gas[ind] * 1e-6), np.log10(sigma_sfr[ind]), nbins = 20, add_last_bin = True)
 
    if(density):
       ind = np.where((sigma_sfr != 0) & (sigma_gas * 1e-6 >= 10**min_gas_dens))
       im = ax.hexbin(np.log10(sigma_gas[ind] * 1e-6), np.log10(sigma_sfr[ind]), xscale='linear', yscale='linear', gridsize=(12,12), cmap='pink_r', mincnt=minct)
    else:
       ind = np.where((sigma_sfr != 0) & (sigma_gas * 1e-6 >= 10**min_gas_dens)  & (third_prop >= vmin) & (third_prop <= vmax))
       im = ax.hexbin(np.log10(sigma_gas[ind] * 1e-6), np.log10(sigma_sfr[ind]), third_prop[ind], gridsize=(10,10), vmin = vmin, vmax = vmax, cmap='pink_r', mincnt=minct, reduce_C_function=np.median)
    ind = np.where(y[0,:] != 0)
    yplot = y[0,ind]
    errdn = y[1,ind]
    errup = y[2,ind]
    ax.errorbar(x[ind],yplot[0], yerr=[yplot[0] - errdn[0], errup[0] - yplot[0]], linestyle='solid', color='k')
    if(save_to_file):
        props_to_save = np.zeros(shape = (len(x[ind]),4))
        props_to_save[:,0] = x[ind]
        props_to_save[:,1] = yplot[0]
        props_to_save[:,2] = yplot[0] - errdn[0]
        props_to_save[:,3] = errup[0] - yplot[0]
        np.savetxt(dir_data + model_name + '/ProcessedData/' + file_name, props_to_save)

    return im

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


def plot_lines_constant_deptime(ax, xmin, xmax, ymaxplot = 10):

    times = [1e8, 1e9, 1e10]
    labels = ['0.1Gyr', '1Gyr', '10Gyr']
    for j in range(0,len(times)):
       ymin = np.log10(10**xmin * 1e6 / times[j])
       ymax = np.log10(10**xmax * 1e6 / times[j])
       if(ymax - 0.15 * (ymax-ymin) < ymaxplot):
          ax.text(xmax - 0.15 * (xmax - xmin), ymax - 0.15 * (ymax-ymin), labels[j], color='grey')
       ax.plot([xmin, xmax], [ymin,ymax], linestyle='dotted', color='grey')

plt = common.load_matplotlib()

def plot_KS_law_obs_only(n0_prof_total, sfr_prof_total, mHI_prof_total, mH2_prof_total, name = 'KS_relation_allgals_z0.pdf', ztarget = 0.0):

    min_gas_dens = -2
    fig = plt.figure(figsize=(14,5))
    xtits = ["$\\rm log_{10} (\\rm \\Sigma_{\\rm HI}/M_{\\odot}\\, pc^{-2})$", "$\\rm log_{10} (\\rm \\Sigma_{\\rm H_{2}}/M_{\\odot}\\, pc^{-2})$", "$\\rm log_{10} (\\rm \\Sigma_{\\rm HI+H_{2}}/M_{\\odot}\\, pc^{-2})$"]
    ytits = ["$\\rm log_{10} (\\rm \\Sigma_{\\rm SFR}/M_{\\odot}\\, yr^{-1}\\, kpc^{-2})$", "",""]
    xmin, xmax, ymin, ymax = min_gas_dens, 2.5, -6, 0.5

    xmin = [0, 0, 0.5]
    xmax = [1.5, 2.5, 2.5]
    ymin = [-5.2,-3.5,-5.5]
    ymax = [-2.5,0.5,-0.5]
    subplots = [131, 132, 133]
   
    for i,s in enumerate(subplots):
        ax = fig.add_subplot(s)
        common.prepare_ax(ax, xmin[i], xmax[i], ymin[i], ymax[i], xtits[i], ytits[i], locators=(0.5, 0.5, 0.5, 0.5))
    
        if i == 0:
            im0 = plot_KS_relation_nogradients(ax,n0_prof_total, sfr_prof_total, mHI_prof_total, min_gas_dens = min_gas_dens)
            plot_HI_obs(ax)
            common.prepare_legend(ax, ['DarkGreen','Teal'], loc = 4)
            plot_lines_constant_deptime(ax, xmin[i], xmax[i], ymaxplot = ymax[i])

        if i == 1:
            im1 = plot_KS_relation_nogradients(ax,n0_prof_total, sfr_prof_total, mH2_prof_total, min_gas_dens = min_gas_dens)
            plot_H2_obs(ax)
            common.prepare_legend(ax, ['Red', 'Red', 'CadetBlue','CornflowerBlue'], loc = 2)
            plot_lines_constant_deptime(ax, xmin[i], xmax[i], ymaxplot = ymax[i])
  
        if i == 2:
            im2 = plot_KS_relation_nogradients(ax,n0_prof_total, sfr_prof_total, mHI_prof_total + mH2_prof_total, min_gas_dens = min_gas_dens)
            plot_Hneutral_obs(ax)
            common.prepare_legend(ax, ['navy','MediumBlue'], loc = 4)
            plot_lines_constant_deptime(ax, xmin[i], xmax[i], ymaxplot = ymax[i])

    common.savefig(outdir, fig, name)


def plot_KS_law(n0_prof_total, sfr_prof_total, mHI_prof_total, mH2_prof_total, prop, prop_name, prop_min = -1, prop_max=2, density = True, name = 'KS_relation_density_allgals_z0.pdf', ztarget = 0.0, save_file = True, file_text=''):

    min_gas_dens = -2
    fig = plt.figure(figsize=(14,6))
    xtits = ["$\\rm log_{10} (\\rm \\Sigma_{\\rm HI}/M_{\\odot}\\, pc^{-2})$", "$\\rm log_{10} (\\rm \\Sigma_{\\rm H_{2}}/M_{\\odot}\\, pc^{-2})$", "$\\rm log_{10} (\\rm \\Sigma_{\\rm HI+H_{2}}/M_{\\odot}\\, pc^{-2})$"]
    ytits = ["$\\rm log_{10} (\\rm \\Sigma_{\\rm SFR}/M_{\\odot}\\, yr^{-1}\\, kpc^{-2})$", "",""]
    xmin, xmax, ymin, ymax = min_gas_dens, 2.5, -6, 0.5
    subplots = [131, 132, 133]
   
    for i,s in enumerate(subplots):
        ax = fig.add_subplot(s)
        common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtits[i], ytits[i], locators=(1, 1, 0.5, 0.5))
    
        if i == 0:
            im0 = plot_KS_relation(ax,n0_prof_total, sfr_prof_total, mHI_prof_total, prop, vmin = prop_min, vmax = prop_max, density = density, min_gas_dens = min_gas_dens, save_to_file = save_file, file_name = 'HISFLaw_z' + str(ztarget) + '_' + method + file_text + '.txt')
            plot_HI_obs(ax)
            common.prepare_legend(ax, ['DarkGreen','Teal'], loc = 2)
            #cbar_ax = fig.add_axes([0.1, 0.9, 0.25, 0.05])
            if(density):
                cbar = fig.colorbar(im0, ax = ax, location = 'top', label = 'Number') #cax=cbar_ax)
            else:
                cbar = fig.colorbar(im0, ax = ax, location = 'top', label = prop_name) #cax=cbar_ax)
            plot_lines_constant_deptime(ax, xmin, xmax)

        if i == 1:
            im1 = plot_KS_relation(ax,n0_prof_total, sfr_prof_total, mH2_prof_total, prop, vmin = prop_min, vmax = prop_max, density = density, min_gas_dens = min_gas_dens, save_to_file = save_file, file_name = 'H2SFLaw_z' + str(ztarget) + '_' + method + file_text + '.txt')
            plot_H2_obs(ax)
            common.prepare_legend(ax, ['Red', 'Red', 'CadetBlue','CornflowerBlue'], loc = 2)
            if(density):
                cbar = fig.colorbar(im1, ax = ax, location = 'top', label = 'Number') #cax=cbar_ax)
            else:
                cbar = fig.colorbar(im1, ax = ax, location = 'top', label = prop_name) #cax=cbar_ax)
            plot_lines_constant_deptime(ax, xmin, xmax)
  
        if i == 2:
            im2 = plot_KS_relation(ax,n0_prof_total, sfr_prof_total, mHI_prof_total + mH2_prof_total, prop, vmin = prop_min, vmax = prop_max, density = density, min_gas_dens = min_gas_dens, save_to_file = save_file, file_name = 'HneutralSFLaw_z' + str(ztarget) + '_' + method + file_text + '.txt')
            plot_Hneutral_obs(ax)
            common.prepare_legend(ax, ['navy','MediumBlue'], loc = 2)
            if(density):
                cbar = fig.colorbar(im2, ax = ax, location = 'top', label = 'Number') #cax=cbar_ax)
            else:
                cbar = fig.colorbar(im2, ax = ax, location = 'top', label = prop_name) #cax=cbar_ax)
            plot_lines_constant_deptime(ax, xmin, xmax)

    common.savefig(outdir, fig, name)

def plot_shi_2011(ax, xmin, xmax):
    xp = np.array([0, 6])

    ax.plot(xp, 1.03 * xp - 4.40, linestyle='solid', color='blue', label='Shi+11')
    ax.plot(xp, 1.03 * xp - 4.40 - 0.08, linestyle='dotted', color='blue')
    ax.plot(xp, 1.03 * xp - 4.40 + 0.08, linestyle='dotted', color='blue')

def plot_ellison_2020(ax, xmin, xmax):
    xp = np.array([0, 6])

    ax.plot(xp, 2.24 * xp - 0.36, linestyle='solid', color='red', label='Ellison+20')
    ax.plot(xp, 2.24 * xp - 0.36 - 0.08, linestyle='dotted', color='red')
    ax.plot(xp, 2.24 * xp - 0.36 + 0.08, linestyle='dotted', color='red')

   
def plot_SigmaGasSigmaStar_law(n0_prof_total, sfr_prof_total, mHI_prof_total, mH2_prof_total, mst_prof_total, name = 'SigmaGasSigmaStar_relation_density_allgals_z0.pdf', ztarget = 0.0, save_file = True, file_text='', density = True):

    min_gas_dens = 1
    fig = plt.figure(figsize=(10,6))
    xtits = ["$\\rm log_{10} (\\rm \\Sigma_{\\rm H_{2}}\,\\Sigma^{0.5}_{\\star}/(M_{\\odot}\\, pc^{-2})^{1.5})$", "$\\rm log_{10} (\\rm \\Sigma_{\\rm HI+H_{2}}\,\\Sigma^{0.5}_{\\star}/(M_{\\odot}\\, pc^{-2})^{1.5})$"]
    ytits = ["$\\rm log_{10} (\\rm \\Sigma_{\\rm SFR}/M_{\\odot}\\, yr^{-1}\\, kpc^{-2})$", ""]
    xmin, xmax, ymin, ymax = -1, 7, -6, 0.5
    subplots = [121, 122]
   
    dummy = np.zeros(shape = len(sfr_prof_total))
    for i,s in enumerate(subplots):
        ax = fig.add_subplot(s)
        common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtits[i], ytits[i], locators=(1, 1, 0.5, 0.5))
    
        if i == 0:
            im1 = plot_KS_relation(ax,n0_prof_total, sfr_prof_total, mH2_prof_total * mst_prof_total, dummy, vmin = xmin, vmax = xmax, density = density, min_gas_dens = min_gas_dens, save_to_file = save_file, file_name = 'H2StarSFLaw_z' + str(ztarget) + '_' + method + file_text + '.txt')
            common.prepare_legend(ax, ['Red', 'Red', 'CadetBlue','CornflowerBlue'], loc = 2)
            if(density):
                cbar = fig.colorbar(im1, ax = ax, location = 'top', label = 'Number') #cax=cbar_ax)
            else:
                cbar = fig.colorbar(im1, ax = ax, location = 'top', label = prop_name) #cax=cbar_ax)
  
        if i == 1:
            im2 = plot_KS_relation(ax,n0_prof_total, sfr_prof_total, (mHI_prof_total + mH2_prof_total) * mst_prof_total, dummy, vmin = xmin, vmax = xmax, density = density, min_gas_dens = min_gas_dens, save_to_file = save_file, file_name = 'HneutralStarSFLaw_z' + str(ztarget) + '_' + method + file_text + '.txt')
            common.prepare_legend(ax, ['navy','MediumBlue'], loc = 2)
            if(density):
                cbar = fig.colorbar(im2, ax = ax, location = 'top', label = 'Number') #cax=cbar_ax)
            else:
                cbar = fig.colorbar(im2, ax = ax, location = 'top', label = prop_name) #cax=cbar_ax)
            plot_shi_2011(ax, xmin, xmax)

    common.savefig(outdir, fig, name)


####### plot KS law for all galaxies ##################

plot_KS_law_obs_only(n0_prof_total, sfr_prof_total, mHI_prof_total, mH2_prof_total, name = 'KS_relation_allgals_'+ ztarget + '_' + method +  '_dr'+ str(dr) + '.pdf', ztarget = ztarget)

plot_KS_law(n0_prof_total, sfr_prof_total, mHI_prof_total, mH2_prof_total, np.log10(mst_prof_total * 1e-6), prop_name = "$\\rm log_{10}(\\Sigma_{\\star}/M_{\\odot}\\,pc^{-2}$)", density = True, name = 'KS_relation_density_allgals_z' + ztarget + '_' + method +  '_dr'+ str(dr) + '.pdf', ztarget = ztarget, save_file = True)

plot_SigmaGasSigmaStar_law(n0_prof_total, sfr_prof_total, mHI_prof_total, mH2_prof_total, mst_prof_total * 1e-6, name = 'SigmaGasSigmaStar_relation_density_allgals_z0.pdf', ztarget = ztarget, save_file = False, file_text='')

plot_KS_law(n0_prof_total, sfr_prof_total, mHI_prof_total, mH2_prof_total, np.log10(gal_prop_ssfr), prop_name = "$\\rm log_{10}(sSFR/yr^{-1})$", prop_min = np.percentile(np.log10(gal_prop_ssfr), 20), prop_max=np.percentile(np.log10(gal_prop_ssfr), 95), density = False, name = 'KS_relation_SSFR_allgals_z' + ztarget + '_' + method +  '_dr'+ str(dr) + '.pdf', ztarget = ztarget, save_file = False)


ind = np.where(mst_prof_total*1e-6 > 10**(-3))  
plot_KS_law(n0_prof_total[ind], sfr_prof_total[ind], mHI_prof_total[ind], mH2_prof_total[ind], np.log10(mst_prof_total[ind] * 1e-6), prop_name = "$\\rm log_{10}(\\Sigma_{\\star}/M_{\\odot}\\,pc^{-2}$)", prop_min = -0.5, prop_max=3, density = False, name = 'KS_relation_SigmaStellarMass_allgals_z' + ztarget + '_' + method +  '_dr'+ str(dr) + '.pdf', ztarget = ztarget, save_file = False)

ind = np.where(oh_prof_total > 0)
plot_KS_law(n0_prof_total[ind], sfr_prof_total[ind], mHI_prof_total[ind], mH2_prof_total[ind], np.log10(oh_prof_total[ind]), prop_name = "$\\rm log_{10}(\\rm O/H$)", prop_min = np.percentile(np.log10(oh_prof_total[ind]), [5]), prop_max=np.percentile(np.log10(oh_prof_total[ind]), [95]), density = False, name = 'KS_relation_OHMetallicity_allgals_z' + ztarget + '_' + method +  '_dr'+ str(dr) + '.pdf', ztarget = ztarget, save_file = False)

ind = np.where(feh_prof_total > 0)
plot_KS_law(n0_prof_total[ind], sfr_prof_total[ind], mHI_prof_total[ind], mH2_prof_total[ind], np.log10(feh_prof_total[ind]), prop_name = "$\\rm log_{10}(\\rm Fe/H$)", prop_min = np.percentile(np.log10(feh_prof_total[ind]), [5]), prop_max=np.percentile(np.log10(feh_prof_total[ind]), [95]), density = False, name = 'KS_relation_FeHMetallicity_allgals_z' + ztarget + '_' + method +  '_dr'+ str(dr) + '.pdf', ztarget = ztarget, save_file = False)

ind = np.where(mdust_prof_total > 0)
plot_KS_law(n0_prof_total[ind], sfr_prof_total[ind], mHI_prof_total[ind], mH2_prof_total[ind], np.log10(mdust_prof_total[ind] * 1e-6), prop_name = "$\\rm log_{10}(\\Sigma_{dust}/M_{\\odot}\\,pc^{-2}$)", prop_min = np.percentile(np.log10(mdust_prof_total[ind] * 1e-6), [5]), prop_max=np.percentile(np.log10(mdust_prof_total[ind] * 1e-6), [95]), density = False, name = 'KS_relation_SigmaDustMass_allgals_z' + ztarget + '_' + method +  '_dr'+ str(dr) + '.pdf', ztarget = ztarget, save_file = False)

plot_KS_law(n0_prof_total, sfr_prof_total, mHI_prof_total, mH2_prof_total, np.log10(gal_prop_mstar), prop_name = "$\\rm log_{10}(\\rm M_{\\star}/M_{\\odot}$)", prop_min = 9, prop_max=10.3, density = False, name = 'KS_relation_TotalStellarMass_allgals_z' + ztarget + '_' + method +  '_dr'+ str(dr) + '.pdf', ztarget = ztarget, save_file = False)

plot_KS_law(n0_prof_total, sfr_prof_total, mHI_prof_total, mH2_prof_total, gal_prop_type, prop_name = "galaxy type", prop_min = 0, prop_max=1, density = False, name = 'KS_relation_GalaxyType_allgals_z' + ztarget + '_' + method +  '_dr'+ str(dr) + '.pdf', ztarget = ztarget, save_file = False)

plot_KS_law(n0_prof_total, sfr_prof_total, mHI_prof_total, mH2_prof_total, gal_prop_disctotot, prop_name = "$\\rm D/T$", prop_min = 0.2, prop_max=1, density = False, name = 'KS_relation_DiskToTotal_allgals_z' + ztarget + '_' + method +  '_dr'+ str(dr) + '.pdf', ztarget = ztarget, save_file = False)

plot_KS_law(n0_prof_total, sfr_prof_total, mHI_prof_total, mH2_prof_total, gal_prop_kappacostar, prop_name = "$\\kappa_{\\star}$", prop_min = 0.1, prop_max=0.8, density = False, name = 'KS_relation_KappacoStars_allgals_z' + ztarget + '_' + method +  '_dr'+ str(dr) + '.pdf', ztarget = ztarget, save_file = False)

plot_KS_law(n0_prof_total, sfr_prof_total, mHI_prof_total, mH2_prof_total, gal_prop_kappacogas, prop_name = "$\\kappa_{\\rm gas}$", prop_min = 0.75, prop_max=1, density = False, name = 'KS_relation_KappacoGas_allgals_z' + ztarget + '_' + method +  '_dr'+ str(dr) + '.pdf', ztarget = ztarget, save_file = False)

plot_KS_law(n0_prof_total, sfr_prof_total, mHI_prof_total, mH2_prof_total, np.log10(gal_prop_r50), prop_name = "$\\rm log_{10}(r_{50}/kpc)$", prop_min = np.percentile(np.log10(gal_prop_r50), [10]), prop_max=np.percentile(np.log10(gal_prop_r50), [90]), density = False, name = 'KS_relation_r50_allgals_z' + ztarget + '_' + method +  '_dr'+ str(dr) + '.pdf', ztarget = ztarget, save_file = False)

plot_KS_law(n0_prof_total, sfr_prof_total, mHI_prof_total, mH2_prof_total, gal_prop_zgas, prop_name = "$\\rm log_{10}(Z_{gas})$", prop_min = np.percentile(gal_prop_zgas, [10]), prop_max=np.percentile(gal_prop_zgas, [90]), density = False, name = 'KS_relation_Zgas_allgals_z' + ztarget + '_' + method +  '_dr'+ str(dr) + '.pdf', ztarget = ztarget, save_file = False)

####### plot KS law for disk galaxies only ############### 
ind = np.where(gal_prop_disctotot > np.median(disctotot))
plot_KS_law(n0_prof_total[ind], sfr_prof_total[ind], mHI_prof_total[ind], mH2_prof_total[ind], np.log10(mst_prof_total[ind] * 1e-6), prop_name = "$\\rm log_{10}(\\Sigma_{\\star}/M_{\\odot}\\,pc^{-2}$)", density = True, name = 'KS_relation_density_diskgals_z' + ztarget + '_' + method + '_dr'+ str(dr) + '.pdf', ztarget = ztarget, save_file = False)

####### plot KW law for satellites/centrals ###############
ind = np.where(gal_prop_type == 0)
if(len(gal_prop_type) > 0):
   plot_KS_law(n0_prof_total[ind], sfr_prof_total[ind], mHI_prof_total[ind], mH2_prof_total[ind], np.log10(mst_prof_total[ind] * 1e-6), prop_name = "$\\rm log_{10}(\\Sigma_{\\star}/M_{\\odot}\\,pc^{-2}$)", density = True, name = 'KS_relation_density_satellites_z' + ztarget + '_' + method + '_dr'+ str(dr) + '.pdf', ztarget = ztarget, save_file = True, file_text = '_satellites')
ind = np.where(gal_prop_type == 1)
plot_KS_law(n0_prof_total[ind], sfr_prof_total[ind], mHI_prof_total[ind], mH2_prof_total[ind], np.log10(mst_prof_total[ind] * 1e-6), prop_name = "$\\rm log_{10}(\\Sigma_{\\star}/M_{\\odot}\\,pc^{-2}$)", density = True, name = 'KS_relation_density_centrals_z' + ztarget + '_' + method + '_dr'+ str(dr) + '.pdf', ztarget = ztarget, save_file = True, file_text = '_centrals')


####### plot KS law for galaxies depending on global SSFR ############### 
ind = np.where(gal_prop_ssfr < ssfr_thresh[0])
plot_KS_law(n0_prof_total[ind], sfr_prof_total[ind], mHI_prof_total[ind], mH2_prof_total[ind], np.log10(mst_prof_total[ind] * 1e-6), prop_name = "$\\rm log_{10}(\\Sigma_{\\star}/M_{\\odot}\\,pc^{-2}$)", density = True, name = 'KS_relation_density_low_ssfr_z' + ztarget + '_' + method + '_dr'+ str(dr) + '.pdf', ztarget = ztarget, save_file = True, file_text = '_ssfr_low')

ind = np.where((gal_prop_ssfr >= ssfr_thresh[0]) & (gal_prop_ssfr <= ssfr_thresh[1]))
plot_KS_law(n0_prof_total[ind], sfr_prof_total[ind], mHI_prof_total[ind], mH2_prof_total[ind], np.log10(mst_prof_total[ind] * 1e-6), prop_name = "$\\rm log_{10}(\\Sigma_{\\star}/M_{\\odot}\\,pc^{-2}$)", density = True, name = 'KS_relation_density_inter_ssfr_z' + ztarget + '_' + method + '_dr'+ str(dr) + '.pdf', ztarget = ztarget, save_file = True, file_text = '_ssfr_inter')

ind = np.where(gal_prop_ssfr > ssfr_thresh[1])
plot_KS_law(n0_prof_total[ind], sfr_prof_total[ind], mHI_prof_total[ind], mH2_prof_total[ind], np.log10(mst_prof_total[ind] * 1e-6), prop_name = "$\\rm log_{10}(\\Sigma_{\\star}/M_{\\odot}\\,pc^{-2}$)", density = True, name = 'KS_relation_density_high_ssfr_z' + ztarget + '_' + method + '_dr'+ str(dr) + '.pdf', ztarget = ztarget, save_file = True, file_text = '_ssfr_high')


################################################## investigate the scatter of the relation ##########################################################

def compute_DeltaSFE(sfr, gas, thresh = 0.0):

    Delta_SFE = np.zeros(shape = len(sfr))
    Delta_SFE[:] = -999 #default value for non well computed things
    toplot = np.where((sfr > 0) & (gas*1e-6 > thresh))
    ##### now compute Delta_SFE using instead a second order polynomial fit to the global KS relation
    ks_power_2nd = np.polyfit(np.log10(gas[toplot] * 1e-6), np.log10(sfr[toplot]), 1)
    x = np.log10(gas[toplot] * 1e-6)
    y = np.log10(sfr[toplot])
    #Delta_SFE[toplot] = y - (ks_power_2nd[0] * x**2 + ks_power_2nd[1] * x + ks_power_2nd[2])
    Delta_SFE[toplot] = y - (ks_power_2nd[0] * x + ks_power_2nd[1])
    return Delta_SFE


def compute_DeltaSFR(sfr, prop, thresh = 0.0):

    Delta_SFR = np.zeros(shape = len(sfr))
    Delta_SFR[:] = -999 #default value for non well computed things
    toplot = np.where((sfr > 0) & (prop > thresh))
    ##### now compute Delta_SFE using instead a second order polynomial fit to the global KS relation
    ks_power_2nd = np.polyfit(prop[toplot], np.log10(sfr[toplot]), 1)
    x = prop[toplot]
    y = np.log10(sfr[toplot])
    #Delta_SFR[toplot] = y - (ks_power_2nd[0] * x**2 + ks_power_2nd[1] * x + ks_power_2nd[2])
    Delta_SFR[toplot] = y - (ks_power_2nd[0] * x + ks_power_2nd[1])

    return Delta_SFR

def plot_delta_relations(ax, Delta_SFR, Delta_ms, Delta_oh, Delta_dust, label = True):

    cols = ['Chocolate', 'DarkGreen', 'DarkMagenta']
    labels = ['$\\Sigma_{\\star}$', '$\\rm log_{10}(O/H)$', '$\\Sigma_{\\rm dust}$']
    for j,c in enumerate(cols):
        if(j == 0):
            prop = Delta_ms
        if(j == 1):
            prop = Delta_oh
        if(j == 2):
            prop = Delta_dust
        ind = np.where((Delta_SFR != -999) & (prop != -999))
        y, x = compute_median_relations(prop[ind], Delta_SFR[ind], nbins = 15, add_last_bin = True)
        ind = np.where(y[0,:] != 0)
        yplot = y[0,ind]
        errdn = y[1,ind]
        errup = y[2,ind]
        ax.fill_between(x[ind], errdn[0], errup[0], facecolor = c, alpha=0.2, interpolate=True)
        ax.plot(x[ind],yplot[0], linestyle='solid', color=c, label = labels[j] if label else None)



ind = np.where(mst_prof_total*1e-6 > 10**(-3))
Delta_SFE_HI = compute_DeltaSFE(sfr_prof_total[ind], mHI_prof_total[ind], thresh = 1)
Delta_SFE_H2 = compute_DeltaSFE(sfr_prof_total[ind], mH2_prof_total[ind], thresh = 1)
Delta_SFE_Hn = compute_DeltaSFE(sfr_prof_total[ind], mHI_prof_total[ind] + mH2_prof_total[ind], thresh = 1)
Delta_ms   = compute_DeltaSFR(sfr_prof_total[ind], np.log10(mst_prof_total[ind]*1e-6), thresh = -0.5)
Delta_oh   = compute_DeltaSFR(sfr_prof_total[ind], oh_prof_total[ind], thresh = -3.0)
Delta_dust = compute_DeltaSFR(sfr_prof_total[ind], np.log10(mdust_prof_total[ind]*1e-6), thresh = -6)


fig = plt.figure(figsize=(6,9))
xtits = ["", "$\\rm \\Delta\\,{SFR-prop}$"]
ytits = ["$\\Delta\\,{\\rm SFE,HI}$", "$\\Delta\\,{\\rm SFE,H_{2}}$","$\\Delta\\,{\\rm SFE,H_{\\rm neutral}}$"]
xmin, xmax, ymin, ymax = -1.5, 3, -0.5, 1.5
subplots = [211, 212]

for i,s in enumerate(subplots):
    ax = fig.add_subplot(s)
    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtits[i], ytits[i], locators=(0.5, 0.5, 0.5, 0.5))

    if i == 0:
        plot_delta_relations(ax, Delta_SFE_HI, Delta_ms, Delta_oh, Delta_dust, label = True)
        common.prepare_legend(ax, ['Chocolate', 'DarkGreen', 'DarkMagenta'], loc = 2)
    if i == 1:
        plot_delta_relations(ax, Delta_SFE_H2, Delta_ms, Delta_oh, Delta_dust, label = False)
        ypmin = -0.04686668613 + 0.5129196387789866 * (-0.6) #fit from Ellison+20
        ypmax = -0.04686668613 + 0.5129196387789866 * 1.1 #fit from Ellison+20
        ax.plot([-0.6,1.1],[ypmin,ypmax], linestyle='dashed', color='Chocolate', label='Ellison+20 fit $\\Sigma_{\\star}$')
        common.prepare_legend(ax, ['Chocolate', 'DarkGreen', 'DarkMagenta'], loc = 2)
    if i == 2:
        plot_delta_relations(ax, Delta_SFE_Hn, Delta_ms, Delta_oh, Delta_dust, label = True)

common.savefig(outdir, fig, "Delta_Correlations_" + method + '_z' + ztarget + 'dr_' + str(dr) + '.pdf')


######################################### check individual KS tracks and fits #######################################################################
def plot_individual_ks_tracks(ax, sfr_prof, gas_prof, thresh = 0.0, plot = True):

    ngals = len(sfr_prof[:,0])
    ks_power = np.zeros(shape = (ngals,2))

    for g in range(0,ngals):
        sfr_in = sfr_prof[g,:]
        gas_in = gas_prof[g,:] 
        toplot = np.where((sfr_in > 0) & (gas_in*1e-6 > thresh))
        if(len(sfr_in[toplot]) > 0):
           if plot == True:
               ax.plot(np.log10(gas_in[toplot] * 1e-6), np.log10(sfr_in[toplot]), linestyle='solid', color='grey', lw = 0.5)
           ks_power[g,:] = np.polyfit(np.log10(gas_in[toplot] * 1e-6), np.log10(sfr_in[toplot]), 1)

    return ks_power

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
        ks_fit_HI = plot_individual_ks_tracks(ax, sfr_prof, mHI_prof, thresh = 1e-2)
        plot_HI_obs(ax)
        common.prepare_legend(ax, ['DarkGreen','Teal'], loc = 2)

    if i == 1:
        ks_fit_H2 = plot_individual_ks_tracks(ax, sfr_prof, mH2_prof)
        ks_fit_H2_highdens = plot_individual_ks_tracks(ax, sfr_prof, mH2_prof, thresh = 2, plot = False)
        plot_H2_obs(ax)
        common.prepare_legend(ax, ['CornflowerBlue','Red', 'Red'], loc = 2)

    if i == 2:
        ks_fit_Hneutral = plot_individual_ks_tracks(ax, sfr_prof, mH2_prof + mHI_prof)
        plot_Hneutral_obs(ax)
        common.prepare_legend(ax, ['navy','MediumBlue'], loc = 2)

common.savefig(outdir, fig, "IndividualTracks_KSlaw_" + method + '_z' + ztarget + 'dr_' + str(dr) + '.pdf')


def plot_ks_best_fit(ax, ks_paramHI, ks_paramH2, ks_paramH2_highdens, ks_paramHn, ymin = 0, ymax = 1.3, label = True, xmin = 0, xmax = 5):

    ind = np.where((ks_paramHI > xmin) & (ks_paramHI < xmax))
    ax.hist(ks_paramHI[ind], bins=15, density = True, color = 'blue', alpha = 0.3, label = 'HI' if label else None)
    ax.plot([np.median(ks_paramHI[ind]), np.median(ks_paramHI[ind])], [ymin, ymax], linestyle='dotted', color = 'blue')
    ind = np.where((ks_paramH2 > xmin) & (ks_paramH2 < xmax))
    ax.hist(ks_paramH2[ind], bins=15, density = True, color = 'red', alpha = 0.3, label = 'H$_2$' if label else None)
    ax.plot([np.median(ks_paramH2[ind]), np.median(ks_paramH2[ind])], [ymin, ymax], linestyle='dotted', color = 'red')
    ind = np.where((ks_paramH2_highdens > xmin) & (ks_paramH2_highdens < xmax))
    ax.hist(ks_paramH2_highdens[ind], bins=15, density = True, color = 'DarkRed', alpha = 0.5, label = 'H$_2$ high-dens' if label else None)
    ax.plot([np.median(ks_paramH2_highdens[ind]), np.median(ks_paramH2_highdens[ind])], [ymin, ymax], linestyle='dotted', color = 'DarkRed')
    ind = np.where((ks_paramHn > xmin) & (ks_paramHn < xmax))
    ax.hist(ks_paramHn[ind], bins=15, density = True, color = 'green', alpha = 0.3, label = 'H neutral' if label else None)
    ax.plot([np.median(ks_paramHn[ind]), np.median(ks_paramHn[ind])], [ymin, ymax], linestyle='dotted', color = 'green')

fig = plt.figure(figsize=(10,6))
xtits = ["$\\rm N_{\\rm KS}$", "$\\rm A_{0}$"]
ytits = ['PDF', '']
ymin, ymax = 0, 1.3
subplots = [121, 122]

xmin = np.array([0,-8])
xmax = np.array([5,0])
for i,s in enumerate(subplots):
    ax = fig.add_subplot(s)
    common.prepare_ax(ax, xmin[i], xmax[i], ymin, ymax, xtits[i], ytits[i], locators=(1, 1, 0.25, 0.25))
    label = False
    if i == 0:
        label = True
    plot_ks_best_fit(ax, ks_fit_HI[:,i], ks_fit_H2[:,i], ks_fit_H2_highdens[:,i], ks_fit_Hneutral[:,i], label = label, xmin = xmin[i], xmax = xmax[i])
    common.prepare_legend(ax, ['blue','red', 'DarkRed', 'green'], loc = 1)
    if i == 1:
        ax.text(-7.9,1.2, "$\\rm \\Sigma_{SFR} = A_0\\, \\Sigma^{N_{KS}}_{gas}$", fontsize=13.5)

common.savefig(outdir, fig, "IndividualFits_KSlaw_" + method + '_z' + ztarget + 'dr_' + str(dr) + '.pdf')

########################### check correlations between galaxy properties and KS fits ############################################################

def plot_med_relations_ks_fits(ax, ks_fit_HI, ks_fit_H2, ks_fit_H2_highdens, ks_fit_Hneutral, third_prop, ymin = 0, ymax = 5, label = True):

    labels = ['HI', '$\\rm H_2$', 'H$_2$ high-dens', 'H neutral']
    cols = ['blue','red', 'DarkRed', 'green']
    for j in range(0,4):
        if j == 0:
            ks_fit = ks_fit_HI
        if j == 1:
            ks_fit = ks_fit_H2
        if j == 2:
            ks_fit = ks_fit_H2_highdens
        if j == 3:
            ks_fit = ks_fit_Hneutral

        ind = np.where((ks_fit > ymin) & (ks_fit < ymax))
        y, x = compute_median_relations(third_prop[ind], ks_fit[ind], nbins = 15, add_last_bin = True)
        ind = np.where(y[0,:] != 0)
        yplot = y[0,ind]
        errdn = y[1,ind]
        errup = y[2,ind]
        ax.fill_between(x[ind], errdn[0], errup[0], facecolor = cols[j], alpha=0.2, interpolate=True)
        ax.plot(x[ind],yplot[0], linestyle='solid', color=cols[j], label = labels[j] if label else None)


def plot_obs_KSfits(ax,slope = True):
    
    ms, A0H2, NHH2, A0all, NHall = np.loadtxt("THINGS_data.txt", usecols = [1, 2, 3, 5, 6])

    NHI, NHIerr, A0HI, A0HIerr, mbHI =  np.loadtxt("Roychowdhury_2009_HIKS.txt", usecols = [3, 4, 5, 6, 7])

    #msHI =  

    

def plot_ks_fits_vs_property(ks_fit_HI, ks_fit_H2, ks_fit_H2_highdens, ks_fit_Hneutral, third_prop, property_name='Stellar Mass', label_prop='$\\rm log_{10}(M_{\\star}/M_{\odot})$', xmin = 9, xmax = 12):
    fig = plt.figure(figsize=(10.8,6))
    ytits = ["$\\rm N_{\\rm KS}$", "$\\rm A_{0}$"]
    subplots = [121, 122]
    
    ymin = np.array([0,-8])
    ymax = np.array([4,0])
    for i,s in enumerate(subplots):
        ax = fig.add_subplot(s)
        common.prepare_ax(ax, xmin, xmax, ymin[i], ymax[i], label_prop, ytits[i], locators=(0.5, 0.5, 0.5, 0.5))
        label = False
        if i == 0:
            label = True
        plot_med_relations_ks_fits(ax, ks_fit_HI[:,i], ks_fit_H2[:,i], ks_fit_H2_highdens[:,i], ks_fit_Hneutral[:,i], third_prop, ymin = ymin[i], ymax = ymax[i], label = label)
        common.prepare_legend(ax, ['blue','red', 'DarkRed', 'green'], loc = 1)
        #if i == 1:
        #    ax.text(-7.9,1.2, "$\\rm \\Sigma_{SFR} = A_0\\, \\Sigma^{N_{KS}}_{gas}$", fontsize=13.5)
   
    common.savefig(outdir, fig, "IndividualFits_KSlaw_Correlation_" + property_name + "_" + method + '_z' + ztarget + 'dr_' + str(dr) + '.pdf')



### now call in plots to check for several correlation ###

plot_ks_fits_vs_property(ks_fit_HI, ks_fit_H2, ks_fit_H2_highdens, ks_fit_Hneutral, np.log10(m30), property_name='TotalStellarMass', label_prop='$\\rm log_{10}(M_{\\star}/M_{\odot})$', xmin = 9, xmax = 11)
plot_ks_fits_vs_property(ks_fit_HI, ks_fit_H2, ks_fit_H2_highdens, ks_fit_Hneutral, np.log10(sfr30/m30), property_name='sSFR', label_prop='$\\rm log_{10}(sSFR/yr^{-1})$', xmin = -12, xmax = -9)
plot_ks_fits_vs_property(ks_fit_HI, ks_fit_H2, ks_fit_H2_highdens, ks_fit_Hneutral, np.log10(mdust), property_name='DustMass', label_prop='$\\rm log_{10}(M_{\\rm dust}/M_{\odot})$', xmin = 5.5, xmax = 8)

print(min(H2frac_galaxy), max(H2frac_galaxy))
plot_ks_fits_vs_property(ks_fit_HI, ks_fit_H2, ks_fit_H2_highdens, ks_fit_Hneutral, np.log10(H2frac_galaxy), property_name='H2frac', label_prop='$\\rm log_{10}(M_{\\rm H_2}/M_{\\star})$', xmin = -2, xmax = 0)

