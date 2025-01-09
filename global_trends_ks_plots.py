import numpy as np
import utilities_statistics as us
import common
import h5py
from hyperfit.linfit import LinFit
from hyperfit.data import ExampleData
plt = common.load_matplotlib()

###################### definition of functions ##########################
def compute_median_relations(x, y, nbins, add_last_bin):
    result, x = us.wmedians_variable_bins(x=x, y=y, nbins=nbins, add_last_bin = add_last_bin)
    return result, x

def plot_KS_relation_nogradients(ax, n0_prof_total, sigma_sfr, sigma_gas, min_gas_dens = -2, color='k', labeln = "", label = False):

    nb = 20
    ind = np.where((sigma_sfr != 0) & (sigma_gas != 0) & (n0_prof_total >= n_thresh_bin))
    if(len(sigma_sfr[ind]) >= 20):
       if(len(sigma_sfr[ind]) < 30):
           nb = 5
       y, x = compute_median_relations(np.log10(sigma_gas[ind] * 1e-6), np.log10(sigma_sfr[ind]), nbins = nb, add_last_bin = True)
       ind = np.where(y[0,:] != 0)
       yplot = y[0,ind]
       errdn = y[1,ind]
       errup = y[2,ind]
       ax.fill_between(x[ind],errdn[0], errup[0], color=color, alpha=0.2)
       ax.plot(x[ind],yplot[0], linestyle='solid', color=color, label = labeln if label == True else None)

def plot_KS_relation(ax, n0_prof_total, sigma_sfr, sigma_gas, third_prop, vmin = -1, vmax=2, density = True, min_gas_dens = -2, save_to_file = False, file_name = 'SFlaw.txt'):

    ind = np.where((sigma_sfr != 0) & (sigma_gas != 0) & (n0_prof_total >= n_thresh_bin))
    if(len(sigma_sfr[ind]) > 0):
       y, x = compute_median_relations(np.log10(sigma_gas[ind] * 1e-6), np.log10(sigma_sfr[ind]), nbins = 20, add_last_bin = True)
 
    if(density):
       ind = np.where((sigma_sfr != 0) & (sigma_gas * 1e-6 >= 10**min_gas_dens))
       if(len(sigma_sfr[ind]) > 0):
          im = ax.hexbin(np.log10(sigma_gas[ind] * 1e-6), np.log10(sigma_sfr[ind]), xscale='linear', yscale='linear', gridsize=(12,12), cmap='pink_r', mincnt=minct)
    else:
       ind = np.where((sigma_sfr != 0) & (sigma_gas * 1e-6 >= 10**min_gas_dens)  & (third_prop >= vmin) & (third_prop <= vmax))
       if(len(sigma_sfr[ind]) > 0):
          im = ax.hexbin(np.log10(sigma_gas[ind] * 1e-6), np.log10(sigma_sfr[ind]), third_prop[ind], gridsize=(10,10), vmin = vmin, vmax = vmax, cmap='pink_r', mincnt=minct, reduce_C_function=np.median)
    if(len(sigma_sfr[ind]) > 0):
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


def plot_selection_by_prop(ax,n0_prof_total, sfr_prof_total, mg_prof_total, prop, prop_name, thresh=[0,1,2], min_gas_dens = -2, label = False):

    ind = np.where(prop < thresh[0])
    if(len(prop[ind]) > 0):
       plot_KS_relation_nogradients(ax,n0_prof_total[ind], sfr_prof_total[ind], mg_prof_total[ind], min_gas_dens = min_gas_dens, color='MediumBlue', labeln=prop_name + '<' + str(thresh[0]), label = label)
    ind = np.where((prop < thresh[1]) & (prop >= thresh[0]))
    if(len(prop[ind]) > 0):
       plot_KS_relation_nogradients(ax,n0_prof_total[ind], sfr_prof_total[ind], mg_prof_total[ind], min_gas_dens = min_gas_dens, color='LimeGreen', labeln=str(thresh[0]) + '<' + prop_name + '<' + str(thresh[1]), label = label)
    if(len(thresh) > 2):
        ind = np.where((prop < thresh[2]) & (prop >= thresh[1]))
        if(len(prop[ind]) > 0):
           plot_KS_relation_nogradients(ax,n0_prof_total[ind], sfr_prof_total[ind], mg_prof_total[ind], min_gas_dens = min_gas_dens, color='red', labeln=str(thresh[1]) + '<' + prop_name + '<' + str(thresh[2]), label = label)
        ind = np.where(prop >= thresh[2])
        if(len(prop[ind]) > 0):
           plot_KS_relation_nogradients(ax,n0_prof_total[ind], sfr_prof_total[ind], mg_prof_total[ind], min_gas_dens = min_gas_dens, color='Maroon', labeln=prop_name + '>' + str(thresh[2]), label = label)
    elif(len(thresh) == 2):
        ind = np.where(prop >= thresh[1])
        if(len(prop[ind]) > 0):
           plot_KS_relation_nogradients(ax,n0_prof_total[ind], sfr_prof_total[ind], mg_prof_total[ind], min_gas_dens = min_gas_dens, color='red', labeln=prop_name + '>' + str(thresh[1]), label = label)



def plot_KS_law_obs_only(n0_prof_total, sfr_prof_total, mHI_prof_total, mH2_prof_total, prop, prop_name, thresh, name = 'KS_relation_allgals_z0.pdf'):

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


def plot_depletion_times_vs_property_deviations(mH2_prof_total, oh_prof_total, mdust_prof_total, mst_prof_total, sfr_prof_total, name_plot='TauDepH2', ylabel = "$\\tau_{\\rm H_{2},annuli}/\\rm Gyr$", density_thresh = 1):
    ###################### compute medians #################################################
    #select the high density regime
    ind = np.where((mH2_prof_total * 1e-6 > density_thresh) & (oh_prof_total > 0) & (mdust_prof_total > 0) & (oh_prof_total < 1e10) & (mdust_prof_total < 1e15) & (mst_prof_total > 0) & (sfr_prof_total > 0))
    yz, xz = compute_median_relations(np.log10(oh_prof_total[ind]), mH2_prof_total[ind]/sfr_prof_total[ind]/1e9, nz, True)
    yd, xd = compute_median_relations(np.log10(mdust_prof_total[ind] * 1e-6), mH2_prof_total[ind]/sfr_prof_total[ind]/1e9, nz, True)
    yssfr, xssfr = compute_median_relations(np.log10(sfr_prof_total[ind]/mst_prof_total[ind]), mH2_prof_total[ind]/sfr_prof_total[ind]/1e9, nz, True)
    ysm, xsm = compute_median_relations(np.log10(mst_prof_total[ind] * 1e-6), mH2_prof_total[ind]/sfr_prof_total[ind]/1e9, nz, True)

    ###################################################################################################################
    ###################### depletion times of individual particles as a function of metallicity ####################################################
    fig = plt.figure(figsize=(5,5))
    ytit = ylabel
    xtit = "$\\rm Property - \\langle Property\\rangle$"
    ymed = np.median(mH2_prof_total[ind]/sfr_prof_total[ind]/1e9)
    xmin, xmax, ymin, ymax = -2, 2, ymed / 20, ymed * 20
   
    ax = fig.add_subplot(111)
    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(0.5,0.5,0.5,0.5))
    ax.set_yscale('log')
    cols = ['Chocolate', 'DarkGreen', 'DarkMagenta', 'MediumBlue']  
    labels = ['$\\rm log_{10}(\\Sigma_{\\star})$', '$\\rm log_{10}(O/H)$', '$\\rm log_{10}(\\Sigma_{\\rm dust})$', '$\\rm log_{10}(\\Sigma_{\\rm sSFR})$']

    for j in range(0,len(cols)):
           if(j == 1):
              xplot = xz[:] - np.median(np.log10(oh_prof_total[ind]))
              yplot = yz[:]
           if(j == 2):
              xplot = xd[:] - np.median(np.log10(mdust_prof_total[ind]) * 1e-6)
              yplot = yd[:]
           if(j == 3):
              xplot = xssfr[:] - np.median(np.log10(sfr_prof_total[ind]/mst_prof_total[ind]))
              yplot = yssfr[:]
           if(j == 0):
              xplot = xsm[:] - np.median(np.log10(mst_prof_total[ind] * 1e-6))
              yplot = ysm[:]
           ymed = yplot[0,:]
           ydn = yplot[1,:]
           yup = yplot[2,:]
           ax.fill_between(xplot, ydn, yup, color=cols[j], alpha=0.2, interpolate=True)
           ax.plot(xplot, ymed, linestyle='solid', color=cols[j] , label=labels[j])
  
    if(name_plot== 'TauDepH2'):
       ax.text(-1.9, ymin + 0.1 * ymin, "Molecular hydrogen", fontsize=12)
    if(name_plot== 'TauDepHI'):
       ax.text(-1.9, ymin + 0.1 * ymin, "Atomic hydrogen", fontsize=12)
    if(name_plot== 'TauDepHn'):
       ax.text(-1.9, ymin + 0.1 * ymin, "Neutral hydrogen", fontsize=12)

    common.prepare_legend(ax, cols, loc = 1)
   
    plt.tight_layout()
   
    common.savefig(outdir, fig, name_plot + '_vs_properties_z' + str(ztarget) + '_' + method + '_dr'+ str(dr) + '.pdf')

    return yz, xz, yd, xd, yssfr, xssfr, ysm, xsm


################################################################# end of function definition #####################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################

 
#define radial bins of interest. This going from 0 to 50kpc, in bins of 1kpc
dir_data = 'Runs/'
model_name = 'L0100N0752/Thermal_non_equilibrium'
#model_name = 'L0025N0376/Thermal_non_equilibrium'
#model_name = 'L0050N0752/Thermal_non_equilibrium'
#choose the type of profile to be read
#method = 'spherical_apertures'
method = 'circular_apertures_face_on_map'
#method = 'circular_apertures_random_map'
##method = 'grid_random_map'
dr = 1.0
zlist = np.array([0.0,1.0,2.0,3.0,4.0,5.0])
#zlist = np.array([0.0, 1.0, 2.0, 3.0])

minct = 3 #minimum number of datapoints per bin to compute median
n_thresh_bin = 10 #minimum number of particles per annuli or bin to consider datapoint in plots

outdir = dir_data + model_name + '/Plots/'


nz = 20
tau_sfr_z = np.zeros(shape = (len(zlist), 3, nz+1))
zbins_sfr_z = np.zeros(shape = (len(zlist), nz+1))
tau_sfr_md = np.zeros(shape = (len(zlist), 3, nz+1))
mdbins_sfr_z = np.zeros(shape = (len(zlist), nz+1))
tau_sfr_ssfr = np.zeros(shape = (len(zlist), 3, nz+1))
ssfrbins_sfr_z = np.zeros(shape = (len(zlist), nz+1))
tau_sfr_sm = np.zeros(shape = (len(zlist), 3, nz+1))
smbins_sfr_z = np.zeros(shape = (len(zlist), nz+1))
tau_sfr_z_hi = np.zeros(shape = (len(zlist), 3, nz+1))
zbins_sfr_z_hi = np.zeros(shape = (len(zlist), nz+1))
tau_sfr_md_hi = np.zeros(shape = (len(zlist), 3, nz+1))
mdbins_sfr_z_hi = np.zeros(shape = (len(zlist), nz+1))
tau_sfr_ssfr_hi = np.zeros(shape = (len(zlist), 3, nz+1))
ssfrbins_sfr_z_hi = np.zeros(shape = (len(zlist), nz+1))
tau_sfr_sm_hi = np.zeros(shape = (len(zlist), 3, nz+1))
smbins_sfr_z_hi = np.zeros(shape = (len(zlist), nz+1))


for s in range(0,len(zlist)):
    ztarget = zlist[s]
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
    sfr_prof = np.loadtxt(dir_data + model_name + '/ProcessedData/' + 'SFR_profiles_ap50ckpc_' + method + '_dr'+ str(dr) + '_z' + str(ztarget) + '.txt')
    mstar_prof = np.loadtxt(dir_data + model_name + '/ProcessedData/' + 'Mstar_profiles_ap50ckpc_' + method + '_dr'+ str(dr) + '_z' + str(ztarget) + '.txt')
    #gas profiles
    mHI_prof = np.loadtxt(dir_data + model_name + '/ProcessedData/' + 'MHI_profiles_ap50ckpc_' + method + '_dr'+ str(dr) + '_z' + str(ztarget) + '.txt')
    mH2_prof = np.loadtxt(dir_data + model_name + '/ProcessedData/' + 'MH2_profiles_ap50ckpc_' + method + '_dr'+ str(dr) + '_z' + str(ztarget) + '.txt')
    mdust_prof = np.loadtxt(dir_data + model_name + '/ProcessedData/' + 'Mdust_profiles_ap50ckpc_' + method + '_dr'+ str(dr) + '_z' + str(ztarget) + '.txt')
    #metallicity profiles
    oh_prof = np.loadtxt(dir_data + model_name + '/ProcessedData/' + 'OH_gas_profiles_ap50ckpc_' + method + '_dr'+ str(dr) + '_z' + str(ztarget) + '.txt')
    feh_prof = np.loadtxt(dir_data + model_name + '/ProcessedData/' + 'FeH_gas_profiles_ap50ckpc_' + method + '_dr'+ str(dr) + '_z' + str(ztarget) + '.txt')
    #number of particles profiles
    n0_prof = np.loadtxt(dir_data + model_name + '/ProcessedData/' + 'NumberPart0_profiles_ap50ckpc_' + method + '_dr'+ str(dr) + '_z' + str(ztarget) + '.txt')
    n4_prof = np.loadtxt(dir_data + model_name + '/ProcessedData/' + 'NumberPart4_profiles_ap50ckpc_' + method + '_dr'+ str(dr) + '_z' + str(ztarget) + '.txt')
    #radial info
    rad_prof = np.loadtxt(dir_data + model_name + '/ProcessedData/' + 'radii_info_' + method + '_dr'+ str(dr) + '_z' + str(ztarget) + '.txt')
    
    
    
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
    
   
    ####### plot KS law for all galaxies ###################################################
   
    ind = np.where(gal_prop_ssfr > 0)
    if(len(gal_prop_ssfr[ind]) > 0):
       plot_KS_law_obs_only(n0_prof_total[ind], sfr_prof_total[ind], mHI_prof_total[ind], mH2_prof_total[ind], np.log10(gal_prop_ssfr[ind]), '$\\rm log_{10}(sSFR)$', [-11, -10, -9.3], name = 'KS_relation_SSFR_trends_z' + str(ztarget) + '_' + method + '_dr'+ str(dr) + '.pdf')
    
    ind = np.where(gal_prop_mstar > 0)
    if(len(gal_prop_ssfr[ind]) > 0):
       plot_KS_law_obs_only(n0_prof_total[ind], sfr_prof_total[ind], mHI_prof_total[ind], mH2_prof_total[ind], np.log10(gal_prop_mstar[ind]), '$\\rm log_{10}(M_{\\star})$', [9.5, 9.9, 10.2], name = 'KS_relation_Mstar_trends_z' + str(ztarget) + '_' + method + '_dr'+ str(dr) + '.pdf')
    
    ind = np.where(sfr_prof_total > 0)
    if(len(gal_prop_ssfr[ind]) > 0):
       plot_KS_law_obs_only(n0_prof_total[ind], sfr_prof_total[ind], mHI_prof_total[ind], mH2_prof_total[ind], gal_prop_kappacostar[ind], '$\\kappa_{\\star}$', [0.2, 0.5], name = 'KS_relation_kappastar_trends_z' + str(ztarget) + '_' + method + '_dr'+ str(dr) + '.pdf')
    
    ind = np.where(gal_prop_mdust > 0)
    if(len(gal_prop_ssfr[ind]) > 0):
       plot_KS_law_obs_only(n0_prof_total[ind], sfr_prof_total[ind], mHI_prof_total[ind], mH2_prof_total[ind], np.log10(gal_prop_mdust[ind]), '$\\rm log_{10}(M_{\\rm dust})$', [6.5, 6.9, 7.3], name = 'KS_relation_Mdust_trends_z' + str(ztarget) + '_' + method + '_dr'+ str(dr) + '.pdf')
    
    ind = np.where(gal_prop_mdust > 0)
    if(len(gal_prop_ssfr[ind]) > 0):
       plot_KS_law_obs_only(n0_prof_total[ind], sfr_prof_total[ind], mHI_prof_total[ind], mH2_prof_total[ind], np.log10(gal_prop_mdust[ind]/gal_prop_mstar[ind]), '$\\rm log_{10}(M_{\\rm dust}/M_{\\star})$', [-3,-2.75,-2.5], name = 'KS_relation_MdustToMstar_trends_z' + str(ztarget) + '_' + method + '_dr'+ str(dr) + '.pdf')


    tau_sfr_z[s,:], zbins_sfr_z[s,:], tau_sfr_md[s,:], mdbins_sfr_z[s,:], tau_sfr_ssfr[s,:], ssfrbins_sfr_z[s,:], tau_sfr_sm[s,:], smbins_sfr_z[s,:] = plot_depletion_times_vs_property_deviations(mH2_prof_total, oh_prof_total, mdust_prof_total, mst_prof_total, sfr_prof_total, name_plot='TauDepH2', ylabel="$\\tau_{\\rm H_2}/\\rm Gyr$", density_thresh = 0.1)
    tau_sfr_z_hi[s,:], zbins_sfr_z_hi[s,:], tau_sfr_md_hi[s,:], mdbins_sfr_z_hi[s,:], tau_sfr_ssfr_hi[s,:], ssfrbins_sfr_z_hi[s,:], tau_sfr_sm_hi[s,:], smbins_sfr_z_hi[s,:] = plot_depletion_times_vs_property_deviations(mHI_prof_total, oh_prof_total, mdust_prof_total, mst_prof_total, sfr_prof_total, name_plot='TauDepHI', ylabel="$\\tau_{\\rm HI}/\\rm Gyr$", density_thresh = 0.1) 
    _, _, _, _, _, _, _, _ = plot_depletion_times_vs_property_deviations(mHI_prof_total + mH2_prof_total, oh_prof_total, mdust_prof_total, mst_prof_total, sfr_prof_total, name_plot='TauDepHn', ylabel="$\\tau_{\\rm HI + H_2}/\\rm Gyr$", density_thresh = 0.1) 


def tau_dep_redshift_evo(tau_sfr_z, zbins_sfr_z, tau_sfr_md, mdbins_sfr_z, tau_sfr_ssfr, ssfrbins_sfr_z, tau_sfr_sm, smbins_sfr_z, name_plot='TauDepH2', ylabel="$\\tau_{\\rm H_2}/\\rm Gyr$", yrange=[0.01,10]):
     ###################################################################################################################
     ###################### depletion times of individual particles as a function of metallicity ####################################################
     fig = plt.figure(figsize=(5,4))
     ytit = ylabel
     xtit = "$\\rm log_{10}(O/H)$"
     xmin, xmax, ymin, ymax = -3, -1, yrange[0], yrange[1]
     
     ax = fig.add_subplot(111)
     common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(0.5,0.5,0.5,0.5))
     ax.set_yscale('log')
     ztoplot=np.array([0.0,1.0,2.0,3.0,4.0,5.0])
     #ztoplot=np.array([0.0,1.0,2.0,3.0])
     #cols = ['Maroon', 'IndianRed', 'red', 'orange', 'gold', 'PaleGreen', 'Navy', 'Indigo']
     cols = ['Maroon', 'red', 'gold', 'green', 'Navy', 'Indigo']
    
     for j in range(0,len(ztoplot)):
         findz = np.where((zlist < ztoplot[j] + 0.01) & (zlist > ztoplot[j] - 0.01))
         if(len(zlist[findz]) > 0):
            xplot = zbins_sfr_z[findz,:]
            yplot = tau_sfr_z[findz,:]
            yplot = yplot[0]
            xplot = xplot[0]
            ymed = yplot[0,0,:]
            ydn = yplot[0,1,:]
            yup = yplot[0,2,:]
            ax.fill_between(xplot[0,:], ydn, yup, color=cols[j], alpha=0.2, interpolate=True)
            ax.plot(xplot[0,:], ymed, linestyle='dotted', color=cols[j] , label='z=%s' % str(ztoplot[j]))
    
     common.prepare_legend(ax, cols, loc = 4)
    
     plt.tight_layout()
    
     common.savefig(outdir, fig, name_plot + '_vs_metallicity_vs_redshift_' + method + '_dr_' + str(dr) + '.pdf')
    
     ###################################################################################################################
     ###################### depletion times of individual particles as a function of dust mass ####################################################
     fig = plt.figure(figsize=(5,4))
     xtit = "$\\rm log_{10}(\\Sigma_{\\rm dust}/M_{\\odot}\\,pc^{-2})$"
     xmin, xmax = -1.5, 1.5
    
     ax = fig.add_subplot(111)
     common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(0.5,0.5,0.5,0.5))
     ax.set_yscale('log')
    
     for j in range(0,len(ztoplot)):
         findz = np.where((zlist < ztoplot[j] + 0.01) & (zlist > ztoplot[j] - 0.01))
         if(len(zlist[findz]) > 0):
            xplot = mdbins_sfr_z[findz,:]
            yplot = tau_sfr_md[findz,:]
            yplot = yplot[0]
            xplot = xplot[0]
            ymed = yplot[0,0,:]
            ydn = yplot[0,1,:]
            yup = yplot[0,2,:]
            ax.fill_between(xplot[0,:], ydn, yup, color=cols[j], alpha=0.2, interpolate=True)
            ax.plot(xplot[0,:], ymed, linestyle='dotted', color=cols[j] , label='z=%s' % str(ztoplot[j]))
     
     common.prepare_legend(ax, cols, loc = 1)
    
     plt.tight_layout()
    
     common.savefig(outdir, fig, name_plot + '_vs_dustmass_vs_redshift_' + method + '_dr_' + str(dr) + '.pdf')
    
     ###################################################################################################################
     ###################### depletion times of individual particles as a function of dust mass ####################################################
     fig = plt.figure(figsize=(5,4))
     xtit = "$\\rm log_{10}(\\Sigma_{\\rm sSFR}/yr^{-1})$"
     xmin, xmax = -11, -7
    
    
     ax = fig.add_subplot(111)
     common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(1,1,0.5,0.5))
     ax.set_yscale('log')
    
     for j in range(0,len(ztoplot)):
         findz = np.where((zlist < ztoplot[j] + 0.01) & (zlist > ztoplot[j] - 0.01))
         if(len(zlist[findz]) > 0):
            xplot = ssfrbins_sfr_z[findz,:]
            yplot = tau_sfr_ssfr[findz,:]
            yplot = yplot[0]
            xplot = xplot[0]
            ymed = yplot[0,0,:]
            ydn = yplot[0,1,:]
            yup = yplot[0,2,:]
            ax.fill_between(xplot[0,:], ydn, yup, color=cols[j], alpha=0.2, interpolate=True)
            ax.plot(xplot[0,:], ymed, linestyle='dotted', color=cols[j] , label='z=%s' % str(ztoplot[j]))
    
     #common.prepare_legend(ax, cols, loc = 3)
    
     plt.tight_layout()
    
     common.savefig(outdir, fig, name_plot + '_vs_ssfr_vs_redshift_' + method + '_dr_' + str(dr) + '.pdf')
    
     ###################################################################################################################
     ###################################################################################################################
     ###################### depletion times of individual particles as a function of dust mass ####################################################
     fig = plt.figure(figsize=(5,4))
     xtit = "$\\rm log_{10}(\\Sigma_{\\star}/M_{\\odot}\\, pc^{-2})$"
     xmin, xmax = -0.5, 3
    
     ax = fig.add_subplot(111)
     common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(1,1,0.5,0.5))
     ax.set_yscale('log')
    
     for j in range(0,len(ztoplot)):
         findz = np.where((zlist < ztoplot[j] + 0.01) & (zlist > ztoplot[j] - 0.01))
         if(len(zlist[findz]) > 0):
            xplot = smbins_sfr_z[findz,:]
            yplot = tau_sfr_sm[findz,:]
            yplot = yplot[0]
            xplot = xplot[0]
            ymed = yplot[0,0,:]
            ydn = yplot[0,1,:]
            yup = yplot[0,2,:]
            ax.fill_between(xplot[0,:], ydn, yup, color=cols[j], alpha=0.2, interpolate=True)
            ax.plot(xplot[0,:], ymed, linestyle='dotted', color=cols[j] , label='z=%s' % str(ztoplot[j]))
    
     #common.prepare_legend(ax, cols, loc = 2)
    
     plt.tight_layout()
    
     common.savefig(outdir, fig, name_plot + '_vs_stellarmass_vs_redshift_' + method + '_dr_' + str(dr) + '.pdf')
    

tau_dep_redshift_evo(tau_sfr_z, zbins_sfr_z, tau_sfr_md, mdbins_sfr_z, tau_sfr_ssfr, ssfrbins_sfr_z, tau_sfr_sm, smbins_sfr_z, name_plot='TauDepH2', ylabel="$\\tau_{\\rm H_2}/\\rm Gyr$", yrange=[0.01,3])
tau_dep_redshift_evo(tau_sfr_z_hi, zbins_sfr_z_hi, tau_sfr_md_hi, mdbins_sfr_z_hi, tau_sfr_ssfr_hi, ssfrbins_sfr_z_hi, tau_sfr_sm_hi, smbins_sfr_z_hi, name_plot='TauDepHI', ylabel="$\\tau_{\\rm HI}/\\rm Gyr$", yrange=[0.1,100])
