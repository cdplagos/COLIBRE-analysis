import numpy as np
import utilities_statistics as us
import common
import h5py
from hyperfit.linfit import LinFit
from hyperfit.data import ExampleData

#define radial bins of interest. This going from 0 to 50kpc, in bins of 1kpc
dir_data = 'Runs/'
model_name = 'L0100N0752/Thermal_non_equilibrium'
#model_name = 'L0025N0376/Thermal_non_equilibrium'
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
gal_prop_mdust = np.zeros(shape = (ngals * nr))

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
        gal_prop_mdust[p] = mdust[g]
        gal_prop_type[p] = typeg[g]
        p = p + 1

def compute_median_relations(x, y, nbins, add_last_bin):
    result, x = us.wmedians_variable_bins(x=x, y=y, nbins=nbins, add_last_bin = add_last_bin)
    return result, x

def plot_KS_relation_nogradients(ax, n0_prof_total, sigma_sfr, sigma_gas, min_gas_dens = -2, color='k', labeln = "", label = False):

    ind = np.where((sigma_sfr != 0) & (sigma_gas != 0) & (n0_prof_total >= n_thresh_bin))
    y, x = compute_median_relations(np.log10(sigma_gas[ind] * 1e-6), np.log10(sigma_sfr[ind]), nbins = 20, add_last_bin = True)
 
    ind = np.where(y[0,:] != 0)
    yplot = y[0,ind]
    errdn = y[1,ind]
    errup = y[2,ind]
    ax.fill_between(x[ind],errdn[0], errup[0], color=color, alpha=0.2)
    ax.plot(x[ind],yplot[0], linestyle='solid', color=color, label = labeln if label == True else None)

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

def plot_selection_by_prop(ax,n0_prof_total, sfr_prof_total, mg_prof_total, prop, prop_name, thresh=[0,1,2], min_gas_dens = -2, label = False):

    ind = np.where(prop < thresh[0])
    plot_KS_relation_nogradients(ax,n0_prof_total[ind], sfr_prof_total[ind], mg_prof_total[ind], min_gas_dens = min_gas_dens, color='MediumBlue', labeln=prop_name + '<' + str(thresh[0]), label = label)
    ind = np.where((prop < thresh[1]) & (prop >= thresh[0]))
    plot_KS_relation_nogradients(ax,n0_prof_total[ind], sfr_prof_total[ind], mg_prof_total[ind], min_gas_dens = min_gas_dens, color='LimeGreen', labeln=str(thresh[0]) + '<' + prop_name + '<' + str(thresh[1]), label = label)
    if(len(thresh) > 2):
        ind = np.where((prop < thresh[2]) & (prop >= thresh[1]))
        plot_KS_relation_nogradients(ax,n0_prof_total[ind], sfr_prof_total[ind], mg_prof_total[ind], min_gas_dens = min_gas_dens, color='red', labeln=str(thresh[1]) + '<' + prop_name + '<' + str(thresh[2]), label = label)
        ind = np.where(prop >= thresh[2])
        plot_KS_relation_nogradients(ax,n0_prof_total[ind], sfr_prof_total[ind], mg_prof_total[ind], min_gas_dens = min_gas_dens, color='Maroon', labeln=prop_name + '>' + str(thresh[2]), label = label)
    elif(len(thresh) == 2):
        ind = np.where(prop >= thresh[1])
        plot_KS_relation_nogradients(ax,n0_prof_total[ind], sfr_prof_total[ind], mg_prof_total[ind], min_gas_dens = min_gas_dens, color='red', labeln=prop_name + '>' + str(thresh[1]), label = label)



def plot_KS_law_obs_only(n0_prof_total, sfr_prof_total, mHI_prof_total, mH2_prof_total, prop, prop_name, thresh, name = 'KS_relation_allgals_z0.pdf', ztarget = 0.0):

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
            plot_selection_by_prop(ax,n0_prof_total, sfr_prof_total, mHI_prof_total, prop, prop_name, thresh, min_gas_dens = min_gas_dens, label = False)
            plot_HI_obs(ax)
            common.prepare_legend(ax, ['DarkGreen','Teal'], loc = 2)
            plot_lines_constant_deptime(ax, xmin[i], xmax[i], ymaxplot = ymax[i])

        if i == 1:
            plot_selection_by_prop(ax,n0_prof_total, sfr_prof_total, mH2_prof_total, prop, prop_name, thresh, min_gas_dens = min_gas_dens, label = False)
            plot_H2_obs(ax)
            common.prepare_legend(ax, ['Red', 'Red', 'CadetBlue','CornflowerBlue'], loc = 2)
            plot_lines_constant_deptime(ax, xmin[i], xmax[i], ymaxplot = ymax[i])
  
        if i == 2:
            plot_selection_by_prop(ax,n0_prof_total, sfr_prof_total, mHI_prof_total + mH2_prof_total, prop, prop_name, thresh, min_gas_dens = min_gas_dens, label = True)
            plot_Hneutral_obs(ax)
            if(len(thresh) > 2):
                common.prepare_legend(ax, ['MediumBlue', 'LimeGreen', 'red','Maroon', 'navy','MediumBlue'], loc = 4)
            elif(len(thresh) == 2):
                common.prepare_legend(ax, ['MediumBlue', 'LimeGreen', 'red', 'navy','MediumBlue'], loc = 4)
            plot_lines_constant_deptime(ax, xmin[i], xmax[i], ymaxplot = ymax[i])

    common.savefig(outdir, fig, name)


####### plot KS law for all galaxies ##################

ind = np.where(gal_prop_ssfr > 0)
plot_KS_law_obs_only(n0_prof_total[ind], sfr_prof_total[ind], mHI_prof_total[ind], mH2_prof_total[ind], np.log10(gal_prop_ssfr[ind]), '$\\rm log_{10}(sSFR)$', [-11, -10, -9.3], name = 'KS_relation_SSFR_trends_z' + ztarget + '_' + method + '_dr'+ str(dr) + '.pdf', ztarget = 0.0)


ind = np.where(gal_prop_mstar > 0)
plot_KS_law_obs_only(n0_prof_total[ind], sfr_prof_total[ind], mHI_prof_total[ind], mH2_prof_total[ind], np.log10(gal_prop_mstar[ind]), '$\\rm log_{10}(M_{\\star})$', [9.5, 9.9, 10.2], name = 'KS_relation_Mstar_trends_z' + ztarget + '_' + method + '_dr'+ str(dr) + '.pdf', ztarget = 0.0)


ind = np.where(sfr_prof_total > 0)
plot_KS_law_obs_only(n0_prof_total[ind], sfr_prof_total[ind], mHI_prof_total[ind], mH2_prof_total[ind], gal_prop_kappacostar[ind], '$\\kappa_{\\star}$', [0.2, 0.5], name = 'KS_relation_kappastar_trends_z' + ztarget + '_' + method + '_dr'+ str(dr) + '.pdf', ztarget = 0.0)

ind = np.where(gal_prop_mdust > 0)
print(len(gal_prop_mdust[ind]))
plot_KS_law_obs_only(n0_prof_total[ind], sfr_prof_total[ind], mHI_prof_total[ind], mH2_prof_total[ind], np.log10(gal_prop_mdust[ind]), '$\\rm log_{10}(M_{\\rm dust})$', [6.5, 6.9, 7.3], name = 'KS_relation_Mdust_trends_z' + ztarget + '_' + method + '_dr'+ str(dr) + '.pdf', ztarget = 0.0)

ind = np.where(gal_prop_mdust > 0)
plot_KS_law_obs_only(n0_prof_total[ind], sfr_prof_total[ind], mHI_prof_total[ind], mH2_prof_total[ind], np.log10(gal_prop_mdust[ind]/gal_prop_mstar[ind]), '$\\rm log_{10}(M_{\\rm dust}/M_{\\star})$', [-3,-2.75,-2.5], name = 'KS_relation_MdustToMstar_trends_z' + ztarget + '_' + method + '_dr'+ str(dr) + '.pdf', ztarget = 0.0)


