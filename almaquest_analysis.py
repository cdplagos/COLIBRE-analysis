#import utilities_statistics as us
import common
import numpy as np

from astropy.io import fits

plt = common.load_matplotlib()

### define constants 
kpc2_to_pc2 = 1e-6
salp_to_cha_stellar_mass = 0.5
salp_to_cha_sfr_halpha = 0.20534/0.1220
###

### define directories and table names
dir_alma = "ALMAQUEST/"

file_pix_data_sfgs = 'H2smsfrsfessfr_sf.cat.fits'
file_pix_data_gvgs = 'H2smsfrsfessfr_gv.cat.fits'
parent_catalogue = 'list_obj_c3c5c6saraall_detect.cat.fits'
###

### read fits data
hdul_sf = fits.open(dir_alma + file_pix_data_sfgs)
hdul_gv = fits.open(dir_alma + file_pix_data_gvgs)
hdul_cat = fits.open(dir_alma + parent_catalogue)

data_sf = hdul_sf[1].data # assuming the first extension is a table
data_gv = hdul_gv[1].data
catalogue = hdul_cat[1].data
### 

ngals = catalogue.shape[0]
#in catalogue we want stellar masses (colum 15) and SFRs (column 16)

sm_cat = np.zeros(shape = ngals)
sfr_cat = np.zeros(shape = ngals)
ssfr_cat = np.zeros(shape = ngals)

for j in range(0,ngals):
   sm_cat[j] = catalogue[j][15] + np.log10(salp_to_cha_stellar_mass)
   sfr_cat[j] = catalogue[j][16] + np.log10(salp_to_cha_sfr_halpha)
   ssfr_cat[j] = catalogue[j][17] + np.log10(salp_to_cha_sfr_halpha) - np.log10(salp_to_cha_stellar_mass) #for a Chabrier IMF

#convert used threshold in Lin+22 to distinguish between SF/GV from a Salpeter to a Chabrier IMF
ssfr_thresh = -10.5 + np.log10(salp_to_cha_sfr_halpha) - np.log10(salp_to_cha_stellar_mass) #for a Chabrier IMF
max_ssfr = max(ssfr_cat)
min_ssfr = min(ssfr_cat)

#now read in data from resolved properties catalogues and convert from a Salpeter to a Chabrier IMF, and from kpc^-2 to pc^-2 for mass surface densities
npixels_gv = data_gv.shape[0]
sigma_H2_gv = np.zeros(shape = npixels_gv)
sigma_sm_gv = np.zeros(shape = npixels_gv)
sigma_sfr_gv = np.zeros(shape = npixels_gv)

npixels_sf = data_sf.shape[0]
sigma_H2_sf = np.zeros(shape = npixels_sf)
sigma_sm_sf = np.zeros(shape = npixels_sf)
sigma_sfr_sf = np.zeros(shape = npixels_sf)

for j in range(0,npixels_gv):
    sigma_H2_gv[j] = data_gv[j][1] + np.log10(kpc2_to_pc2)
    sigma_sm_gv[j] = data_gv[j][2] + np.log10(salp_to_cha_stellar_mass) + np.log10(kpc2_to_pc2)
    sigma_sfr_gv[j] = data_gv[j][3] + np.log10(salp_to_cha_sfr_halpha) 

for j in range(0,npixels_sf):
    sigma_H2_sf[j] = data_sf[j][1] + np.log10(kpc2_to_pc2)
    sigma_sm_sf[j] = data_sf[j][2] + np.log10(salp_to_cha_stellar_mass) + np.log10(kpc2_to_pc2)
    sigma_sfr_sf[j] = data_sf[j][3] + np.log10(salp_to_cha_sfr_halpha) 

hdul_sf.close()
hdul_gv.close()
hdul_cat.close()

##### now read z=0 catalogue to select galaxies with the same stellar mass and SSFR distribution ##########

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
ztarget = '0.0'

minct = 3 #minimum number of datapoints per bin to compute median
n_thresh_bin = 10 #minimum number of particles per annuli or bin to consider datapoint in plots

outdir = dir_data + model_name + '/Plots/'

data = np.loadtxt(dir_data + model_name + '/ProcessedData/' + 'GalaxyProperties_z' + str(ztarget) + '.txt')
idgal        = data[:,0]
m30          = data[:,5]
sfr30        = data[:,6]
ssfr30 = sfr30/m30
ind = np.where(ssfr30 == 0)
ssfr30[ind] = 1e-10
ssfr30 = np.log10(ssfr30)
m30 = np.log10(m30)
ngals = len(idgal)

### now read profiles #####

#sfr and stellar mass profiles
sfr_prof = np.loadtxt(dir_data + model_name + '/ProcessedData/' + 'SFR_profiles_ap50ckpc_' + method + '_dr'+ str(dr) + '_z' + ztarget + '.txt')
print(sfr_prof.shape)
mstar_prof = np.loadtxt(dir_data + model_name + '/ProcessedData/' + 'Mstar_profiles_ap50ckpc_' + method + '_dr'+ str(dr) + '_z' + ztarget + '.txt')
#gas profiles
mH2_prof = np.loadtxt(dir_data + model_name + '/ProcessedData/' + 'MH2_profiles_ap50ckpc_' + method + '_dr'+ str(dr) + '_z' + ztarget + '.txt')
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
        mstar_prof[g,:] = mstar_prof[g,:] / area_annuli[:] * kpc2_to_pc2
        mH2_prof[g,:] = mH2_prof[g,:] / area_annuli[:] * kpc2_to_pc2

elif (method == 'grid_random_map'):
    nr = len(rad_prof)
    area_pixel = dr**2
    for g in range(0,ngals):
        sfr_prof[g,:] = sfr_prof[g,:] / area_pixel
        mstar_prof[g,:] = mstar_prof[g,:] / area_pixel * kpc2_to_pc2
        mH2_prof[g,:] = mH2_prof[g,:] / area_pixel * kpc2_to_pc2


### select galaxies in COLIBRE ####

def select_mass_matched_sample(gv = True, boost = 10):
     mmin = 9.5
     mmax = 12.5
     dm = 0.15
     mbins = np.arange(mmin, mmax, dm)
     xm = mbins + dm/2.0
     
     if gv == True:
         ind = np.where(ssfr_cat <= ssfr_thresh) 
     else:
         ind = np.where(ssfr_cat > ssfr_thresh)
     print("selected galaxies in ALMAQUEST", len(sm_cat[ind]))
     H, _ = np.histogram(sm_cat[ind], bins=np.append(mbins,mmax))
     
     ids_selected = np.zeros(shape = (len(sm_cat[ind]) * boost))
     ids_selected[:] = -99
     idgal_ordered = np.arange(0,len(m30),1)
     p = 0
     for i,m in enumerate(xm):
         n_cat_bin = H[i] * boost
         if(n_cat_bin > 0):
            if gv == True:
                ind = np.where((m30 > m - dm/2.0) & (m30 <= m + dm/2.0) & (ssfr30 <= ssfr_thresh) & (ssfr30 >= min_ssfr))
            else:
                ind = np.where((m30 > m - dm/2.0) & (m30 <= m + dm/2.0) & (ssfr30 > ssfr_thresh) & (ssfr30 <= max_ssfr))
            nselected = len(m30[ind])
            print(nselected, n_cat_bin)
            if(nselected > n_cat_bin):
               nbin = n_cat_bin
               ids_selected[p:p+nbin] = np.random.choice(idgal_ordered[ind], size=nbin)
            else:
               nbin = nselected
               ids_selected[p:p+nbin] = idgal_ordered[ind]
            p = p + nbin
   
     ind = np.where(ids_selected >= 0)
     ids_selected = ids_selected[ind]

     nin = len(ids_selected)
     print("COLIBRE selected to mass match", nin)
     sfr_prof_total = np.zeros(shape = (nin * nr))
     mst_prof_total = np.zeros(shape = (nin * nr))
     mH2_prof_total = np.zeros(shape = (nin * nr))
     n0_prof_total = np.zeros(shape = (nin * nr))
     p = 0
     for g in range(0,nin):
         for r in range(0,nr):
             ind = np.where(idgal_ordered == ids_selected[g])
             if(len(idgal_ordered[ind]) > 0):
                sfr_prof_total[p] = sfr_prof[g,r]
                mst_prof_total[p] = mstar_prof[g,r]
                mH2_prof_total[p] = mH2_prof[g,r]
                n0_prof_total[p] = n0_prof[g,r]
                p = p + 1

     return mH2_prof_total, sfr_prof_total, mst_prof_total, n0_prof_total

mh2_gv, sfr_gv, ms_gv, n0_gv = select_mass_matched_sample(gv = True, boost = 14)
mh2_sf, sfr_sf, ms_sf, n0_sf = select_mass_matched_sample(gv = False, boost = 8)


############### now plot ###################################################
def plot_KS_relation_nogradients(ax, n0_prof_total, sigma_sfr, sigma_gas, sigma_sfr_obs, sigma_gas_obs, min_gas_dens = -2, gv = True):

    ind = np.where((sigma_sfr != 0) & (sigma_gas > 10**min_gas_dens) & (n0_prof_total >= n_thresh_bin))
    y, x = compute_median_relations(np.log10(sigma_gas[ind] * 1e-6), np.log10(sigma_sfr[ind]), nbins = 20, add_last_bin = True)

    label = 'COLIBRE SF-like'
    obslabel = 'ALMAQUEST SF'
    if(gv == True):
        label = 'COLIBRE GV-like'
        obslabel = 'ALMAQUEST GV'

    ind = np.where(y[0,:] != 0)
    yplot = y[0,ind]
    errdn = y[1,ind]
    errup = y[2,ind]
    ax.fill_between(x[ind],errdn[0], errup[0], color='k', alpha=0.2)

    ax.plot(x[ind],yplot[0], linestyle='solid', color='k', label = label)

    ind = np.where((sigma_sfr_obs != 0) & (sigma_gas_obs != 0))
    y, x = compute_median_relations(np.log10(sigma_gas_obs[ind]), np.log10(sigma_sfr_obs[ind]), nbins = 20, add_last_bin = True)

    ind = np.where(y[0,:] != 0)
    yplot = y[0,ind]
    errdn = y[1,ind]
    errup = y[2,ind]
    ax.errorbar(x[ind],yplot[0],yerr=[yplot[0] - errdn[0], errup[0] - yplot[0]], color='red', marker='S', label = obslabel)


def plot_lines_constant_deptime(ax, xmin, xmax, ymaxplot = 10):

    times = [1e8, 1e9, 1e10]
    labels = ['0.1Gyr', '1Gyr', '10Gyr']
    for j in range(0,len(times)):
       ymin = np.log10(10**xmin * 1e6 / times[j])
       ymax = np.log10(10**xmax * 1e6 / times[j])
       if(ymax - 0.15 * (ymax-ymin) < ymaxplot):
          ax.text(xmax - 0.15 * (xmax - xmin), ymax - 0.15 * (ymax-ymin), labels[j], color='grey')
       ax.plot([xmin, xmax], [ymin,ymax], linestyle='dotted', color='grey')


print("Will produce plot now...")
fig = plt.figure(figsize=(5,8))
print("will define axes")
xtit = "$\\rm log_{10} (\\rm \\Sigma_{\\rm H_{2}}/M_{\\odot}\\, pc^{-2})$",
ytit = "$\\rm log_{10} (\\rm \\Sigma_{\\rm SFR}/M_{\\odot}\\, yr^{-1}\\, kpc^{-2})$"
xmin, xmax, ymin, ymax = 0, 2.5, -6, 0.5

subplots = [121, 122]

print("will loop over subplots")
for i,s in enumerate(subplots):
        ax = fig.add_subplot(s)
        common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(0.5, 0.5, 0.5, 0.5))
        
        if i == 0:
            print("will plot SF galaxies")
            im0 = plot_KS_relation_nogradients(ax,n0_sf, sfr_sf, mH2_sf, sigma_H2_sf, sigma_sfr_sf, min_gas_dens = 0)
            common.prepare_legend(ax, ['red','Teal'], loc = 4)
            plot_lines_constant_deptime(ax, xmin, xmax, ymaxplot = ymax)
        
        if i == 1:
            print("will plot GV galaxies")
            im0 = plot_KS_relation_nogradients(ax,n0_gv, sfr_gv, mH2_gv, sigma_H2_gv, sigma_sfr_gv, min_gas_dens = 0)
            common.prepare_legend(ax, ['Red', 'Red', 'CadetBlue','CornflowerBlue'], loc = 2)
            plot_lines_constant_deptime(ax, xmin, xmax, ymaxplot = ymax)

common.savefig(outdir, fig, "ALMAQUEST_matched_KS.pdf")


