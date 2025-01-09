import numpy as np
import utilities_statistics as us 
import common
import h5py
import scipy.optimize as so

def compute_median_relations(x, y, nbins, add_last_bin):
    result, x = us.wmedians_variable_bins(x=x, y=y, nbins=nbins, add_last_bin = add_last_bin)
    return result, x


def plot_ISM_phases(ax, xmin, xmax):

    T1 = 4.5
    T2 = 3
    ax.plot([xmin, xmax], [T1, T1], linestyle='dotted', color='gray')
    ax.plot([xmin, xmax], [T2, T2], linestyle='dotted', color='gray')
    ax.text(xmax - 0.1 * (xmax-xmin), T1 + 0.2, 'HIM', color='gray')
    ax.text(xmax - 0.1 * (xmax-xmin), T1 - 0.4, 'WNM', color='gray')
    ax.text(xmax - 0.1 * (xmax-xmin), T2 - 0.4, 'CNM', color='gray')


def find_confidence_interval(x, pdf, confidence_level):
    return pdf[pdf > x].sum() - confidence_level


def density_contour(ax, xdata, ydata, nbins_x, nbins_y, cmap = 'grey'):
    """ Create a density contour plot.
    Parameters
    ----------
    ax : matplotlib.Axes
        Plot the contour to this axis
    xdata : numpy.ndarray
    ydata : numpy.ndarray
    nbins_x : int
        Number of bins along x dimension
    nbins_y : int
        Number of bins along y dimension
    contour_kwargs : dict
        kwargs to be passed to pyplot.contour()
    """

    H, xedges, yedges = np.histogram2d(xdata, ydata, bins=(nbins_x,nbins_y), density=True)
    x_bin_sizes = (xedges[1:] - xedges[:-1]).reshape((1,nbins_x))
    y_bin_sizes = (yedges[1:] - yedges[:-1]).reshape((nbins_y,1))

    pdf = (H*(x_bin_sizes*y_bin_sizes))

    thirty_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.5))
    one_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.68))
    two_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.95))
    three_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.99))
    levels = [three_sigma, two_sigma, one_sigma, thirty_sigma]


    X, Y = 0.5*(xedges[1:]+xedges[:-1]), 0.5*(yedges[1:]+yedges[:-1])
    Z = pdf.T

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.colors as col

    # The viridis colormap is only available since mpl 1.5
    extra_args = {}
    if tuple(mpl.__version__.split('.')) >= ('1', '5'):
        extra_args['cmap'] = plt.get_cmap(cmap)

    return ax.contour(X, Y, Z, levels=levels, origin="lower", alpha=1,
                      norm=col.Normalize(vmin=0, vmax=0.025), **extra_args)

#######################################################################################
dir_data = 'Runs/'
model_name =  'L0025N0376/Thermal_non_equilibrium'

ztarget = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 3.5, 4.0, 4.5, 5.0, 5.5])
plots = [True, False, False, False, False, False, False, False, False, False, True, False, False, False, True, True, True, True, True, True]

outdir = dir_data + model_name + '/Plots/'
plt = common.load_matplotlib()

zmax = -0.1
zmin = -4
dz = 1.0
zbins = np.arange(zmin, zmax, dz)
xz = zbins + dz/2.0

nmin = -2
nmax = 4
dn = 0.2
nhbins = np.arange(nmin, nmax, dn)
xn = nhbins + dn/2.0


frac_sfr = np.zeros(shape = (4, len(ztarget)))
frac_sfr_z = np.zeros(shape = (2, len(ztarget), len(zbins)))

frac_hi = np.zeros(shape = (4, len(ztarget)))
frac_h2 = np.zeros(shape = (4, len(ztarget)))

frac_wnm = np.zeros(shape = (2, len(ztarget)))
frac_cnm = np.zeros(shape = (2, len(ztarget)))

dens_cnm = np.zeros(shape = (3, len(ztarget)))
dens_cnm_w = np.zeros(shape = (3, 3, len(ztarget)))

med_tauh2 = np.zeros(shape = (3,len(ztarget)))
med_tauhi = np.zeros(shape = (3,len(ztarget)))

nz = 20
tau_sfr_z = np.zeros(shape = (len(ztarget), 3, nz+1))
zbins_sfr_z = np.zeros(shape = (len(ztarget), nz+1))
tau_sfr_md = np.zeros(shape = (len(ztarget), 3, nz+1))
mdbins_sfr_z = np.zeros(shape = (len(ztarget), nz+1))


hist_cnm_h2 = np.zeros(shape = (len(ztarget), len(xn)))
hist_cnm_hi = np.zeros(shape = (len(ztarget), len(xn)))
hist_cnm_sfr = np.zeros(shape = (len(ztarget), len(xn)))

for i,z in enumerate(ztarget):
     print("Will process redshift ", z)
     data = np.loadtxt(dir_data + model_name + '/ProcessedData/' + 'particles_ap50ckpc_z' + str(z) + '.txt')
   
     mHI = data[0,:]
     mH2 = data[1,:]
     T = data[2,:]
     dens = data[3,:]
     sfr = data[4,:]
     zgas = data[5,:]
     mgas = data[6,:]
     mdust = data[7,:]

     T = np.log10(T)
     dens = np.log10(dens)

     for j in range(0,len(zbins)):
         ind = np.where((np.log10(zgas) >= xz[j] - dz/2.0) & (np.log10(zgas) < xz[j] + dz/2.0) & (sfr > 0))
         if(len(sfr[ind]) > 0):
             sfr_in_z = sfr[ind]
             T_in = T[ind]
             cnm = np.where(T_in < np.log10(1000))
             frac_sfr_z[0,i,j] = sum(sfr_in_z[cnm])/sum(sfr_in_z)
             wnm = np.where((T_in < 4.8) & (T_in >= np.log10(1000)))
             frac_sfr_z[1,i,j] = sum(sfr_in_z[wnm])/sum(sfr_in_z)

     ind = np.where(sfr > 0)
     lowper = np.percentile(sfr[ind], [10])
     ind = np.where((sfr > lowper) & (mH2 > 0) & (mHI > 0))
     tau_sfr_z[i,:], zbins_sfr_z[i,:] = compute_median_relations(np.log10(zgas[ind]), mH2[ind]/sfr[ind]/1e9, nz, True)
     tau_sfr_md[i,:], mdbins_sfr_z[i,:] = compute_median_relations(np.log10(mdust[ind]), mH2[ind]/sfr[ind]/1e9, nz, True)

     tau_h2 = mH2[ind]/sfr[ind]/1e9 #Gyr
     tau_hi = mHI[ind]/sfr[ind]/1e9 #Gyr

     perh2 = np.percentile(tau_h2, [16,50,84])
     perhi = np.percentile(tau_hi, [16,50,84])

     med_tauh2[:,i] = perh2
     med_tauhi[:,i] = perhi

     ind = np.where((T < 4.8) & (T > np.log10(1000)))
     frac_wnm[0,i] = np.sum(mHI[ind]) / np.sum(mgas[ind])
     frac_wnm[1,i] = np.sum(mH2[ind]) / np.sum(mgas[ind])

     ind = np.where(T < np.log10(1000))
     frac_cnm[0,i] = np.sum(mHI[ind]) / np.sum(mgas[ind])
     frac_cnm[1,i] = np.sum(mH2[ind]) / np.sum(mgas[ind])
  
     H, _ = np.histogram(dens[ind], weights = mH2[ind] / sum(mH2), bins=np.append(nhbins,nmax))
     hist_cnm_h2[i,:] =  H
     H, _ = np.histogram(dens[ind], weights = mHI[ind] / sum(mHI), bins=np.append(nhbins,nmax))
     hist_cnm_hi[i,:] =  H
     H, _ = np.histogram(dens[ind], weights = sfr[ind] / sum(sfr), bins=np.append(nhbins,nmax))
     hist_cnm_sfr[i,:] =  H

     dens_cnm[0,i] = np.log10(sum(10**dens[ind] * mH2[ind]) / sum(mH2[ind]))
     dens_cnm[1,i] = np.log10(sum(10**dens[ind] * mHI[ind]) / sum(mHI[ind]))
     dens_cnm[2,i] = np.log10(sum(10**dens[ind] * sfr[ind]) / sum(sfr[ind]))
   
     dens_cnm_w[0,:,i] = us.weighted_quantile(dens[ind], np.array([0.25,0.50,0.75]), sample_weight=mH2[ind]) #, old_style=True)
     #print(dens_cnm[0,i], dens_cnm_w[0,1,i]) 
     dens_cnm_w[1,:,i] = us.weighted_quantile(dens[ind], np.array([0.25,0.50,0.75]), sample_weight=mHI[ind])
     dens_cnm_w[2,:,i] = us.weighted_quantile(dens[ind], np.array([0.25,0.50,0.75]), sample_weight=sfr[ind])

     #now normalise masses and sFRs
     mHI = mHI / sum(mHI)
     mH2 = mH2 / sum(mH2)
     sfr = sfr / sum(sfr)
    
     ind = np.where((T > 4.8) & (dens < -1))
     frac_sfr[0,i] = np.sum(sfr[ind])
     frac_h2[0,i] = np.sum(mH2[ind])
     frac_hi[0,i] = np.sum(mHI[ind])
     
     ind = np.where((T < 4.8) & (T > np.log10(1000)))
     frac_sfr[1,i] = np.sum(sfr[ind])
     frac_h2[1,i] = np.sum(mH2[ind])
     frac_hi[1,i] = np.sum(mHI[ind])
    
     ind = np.where(T < np.log10(1000))
     frac_sfr[2,i] = np.sum(sfr[ind])
     frac_h2[2,i] = np.sum(mH2[ind])
     frac_hi[2,i] = np.sum(mHI[ind])

     ind = np.where((T < 4.1) & (T>3.9) & (dens > 1))
     frac_sfr[3,i] = np.sum(sfr[ind])
     frac_h2[3,i] = np.sum(mH2[ind])
     frac_hi[3,i] = np.sum(mHI[ind])
 
     print("fraction of SFR in HII regions", frac_sfr[3,i], " at redshift", z)
     if(plots[i] == True):
        ####################### plot all methods at z=0 #########################################
        min_gas_dens = -2
        fig = plt.figure(figsize=(5,12))
        xtits = ['', '', "$\\rm log_{10} (\\rm n_{\\rm H}/cm^{-3})$"]
        ytit = "$\\rm log_{10} (\\rm T/K)$"
        xtit = "$\\rm log_{10} (\\rm n_{\\rm H}/cm^{-3})$"
       
        xmin, xmax, ymin, ymax = -4, 4, 1, 7
        subplots = [311, 312, 313]
        xtext = xmax - 0.4 * (xmax - xmin)
        ytext = ymax - 0.1 * (ymax - ymin)
        
        for j,s in enumerate(subplots):
            ax = fig.add_subplot(s)
            common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(1,1,1,1))
        
            if j == 0:
                im = ax.hexbin(dens, T, mHI, xscale='linear', yscale='linear', gridsize=(20,20), cmap='Blues', mincnt=1, reduce_C_function = np.sum, bins = 'log', vmin = 1e-6, vmax = 0.1)
                cbar = fig.colorbar(im, ax = ax, location = 'top', label = 'HI fraction')
                density_contour(ax, dens, T, 30, 30, cmap = 'Purples_r')
                ax.text(xtext, ytext, 'Atomic hygroden')
                plot_ISM_phases(ax, xmin, xmax)
            if j == 1:
                im = ax.hexbin(dens, T, mH2, xscale='linear', yscale='linear', gridsize=(20,20), cmap='Oranges', mincnt=1, reduce_C_function = np.sum, bins = 'log', vmin = 1e-6, vmax = 0.1)
                cbar = fig.colorbar(im, ax = ax, location = 'top', label = 'H$_{2}$ fraction')
                density_contour(ax, dens, T, 30, 30, cmap = 'Purples_r')
                ax.text(xtext, ytext, 'Molecular hygroden')
                plot_ISM_phases(ax, xmin, xmax)
            if j == 2:
                im = ax.hexbin(dens, T, sfr, xscale='linear', yscale='linear', gridsize=(20,20), cmap='Greens', mincnt=1, reduce_C_function = np.sum, bins = 'log', vmin = 1e-6, vmax = 0.1)
                cbar = fig.colorbar(im, ax = ax, location = 'top', label = 'SFR fraction')
                density_contour(ax, dens, T, 30, 30, cmap = 'Purples_r')
                ax.text(xtext, ytext, 'SFR')
                plot_ISM_phases(ax, xmin, xmax)
       
        plt.tight_layout()
       
        common.savefig(outdir, fig, 'Particles_PhaseSpace_HIH2_z' + str(z) + '.pdf')
   
        ####################### plot all methods at z=0 #########################################
        min_gas_dens = -2
        fig = plt.figure(figsize=(5,4))
        ytit = "density"
        xtit = "$\\rm log_{10} (\\rm n_{\\rm H,CNM}/cm^{-3})$"
       
        xmin, xmax, ymin, ymax = -2, 4, 0, 1
        ax = fig.add_subplot(111)
        common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(1,1,0.2,0.2))
        ax.plot(xn, hist_cnm_h2[i,:]/sum(hist_cnm_h2[i,:] * dn), linestyle='dashed', color='red', label='H$_2$-weighted')
        ax.plot(xn, hist_cnm_hi[i,:]/sum(hist_cnm_hi[i,:] * dn), linestyle='dashed', color='blue', label='HI-weighted')
        ax.plot(xn, hist_cnm_sfr[i,:]/sum(hist_cnm_sfr[i,:] * dn), linestyle='dashed', color='green', label='SFR-weighted')
        common.prepare_legend(ax, ['red','blue', 'green'], loc = 2)

        plt.tight_layout()
        print("will save figure", outdir + 'Particles_densityCNM_HIH2_z' + str(z) + '.pdf')
        common.savefig(outdir, fig, 'Particles_densityCNM_HIH2_z' + str(z) + '.pdf')
   


############################# mass fraction locked up in different ISM phases ##############################
fig = plt.figure(figsize=(6,12))
ytit = "Fraction of gas phases in WNM/CNM"
xtit = "redshift"
xmin, xmax, ymin, ymax = min(ztarget), max(ztarget), -0.1, 1.1
        
ax = fig.add_subplot(311)
common.prepare_ax(ax, xmin, xmax, ymin, ymax, " ", ytit, locators=(1,1,0.1,0.1))

#ax.plot(ztarget, frac_hi[0,:], linestyle='dotted', color='blue')
ax.plot(ztarget, frac_hi[1,:], linestyle='solid', color='blue', label = 'HI')
ax.plot(ztarget, frac_hi[2,:], linestyle='dashed', color='blue')

#ax.plot(ztarget, frac_h2[0,:], linestyle='dotted', color='red')
ax.plot(ztarget, frac_h2[1,:], linestyle='solid', color='red' , label = 'H$_2$')
ax.plot(ztarget, frac_h2[2,:], linestyle='dashed', color='red')

#ax.plot(ztarget, frac_sfr[0,:], linestyle='dotted', color='green' )
ax.plot(ztarget, frac_sfr[1,:], linestyle='solid', color='green', label = 'SFR')
ax.plot(ztarget, frac_sfr[2,:], linestyle='dashed', color='green')

ax.text(0.1,1.02,'H$_2$ in CNM', color='red')
ax.text(0.1,0.87,'SFR in CNM', color='green')
ax.text(0.1,0.67,'HI in WNM', color='blue')

ax.text(0.1,-0.05,'H$_2$ in WNM', color='red')
ax.text(0.1,0.1,'SFR in WNM', color='green')
ax.text(0.1,0.39,'HI in CNM', color='blue')


#ax.plot([2.5,3],[0.56,0.56],linestyle='solid',color='grey')
#ax.text(3.1,0.56,'WNM',color='grey')
#ax.plot([2.5,3],[0.5,0.5],linestyle='dashed',color='grey')
#ax.text(3.1,0.5,'CNM',color='grey')
#common.prepare_legend(ax, ['blue','red', 'green'], loc = 7)

ytit = "Fraction of WNM/CNM in gas phases"
ax = fig.add_subplot(312)
ymax = 0.65
common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(1,1,0.1,0.1))

ax.plot(ztarget, frac_wnm[0,:], linestyle='dashed', color='k', label='WNM HI')
ax.plot(ztarget, frac_wnm[1,:], linestyle='solid', color='k', label = 'WNM H$_2$')

ax.plot(ztarget, frac_cnm[0,:], linestyle='dashed', color='darkorange', label='CNM HI')
ax.plot(ztarget, frac_cnm[1,:], linestyle='solid', color='darkorange', label = 'CNM H$_2$')

#common.prepare_legend(ax, ['k','k', 'darkorange', 'darkorange'], loc = 7)
ax.text(0.1,0.62,'WNM in HI', color='k')
ax.text(0.1,0.48,'CNM in HI', color='darkorange')

ax.text(0.1,-0.05,'WNM in H$_2$', color='k')
ax.text(0.1,0.2,'CNM in H$_2$', color='darkorange')


ytit = "$\\rm log_{10}(n_{\\rm H,CNM}/cm^{-3})$"
ax = fig.add_subplot(313)
ymin, ymax = -1.5, 2.5
common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(1,1,0.5,0.5))

#ax.plot(ztarget, dens_cnm[0,:], linestyle='dashed', color='red', label='H$_2$-weighted')
ax.fill_between(ztarget, dens_cnm_w[0,0,:],  dens_cnm_w[0,2,:], facecolor='red', alpha=0.2)
ax.plot(ztarget, dens_cnm_w[0,1,:], linestyle='dashed', color='red', label='H$_2$-weighted')

ax.fill_between(ztarget, dens_cnm_w[1,0,:],  dens_cnm_w[1,2,:], facecolor='blue', alpha=0.2)
ax.plot(ztarget, dens_cnm_w[1,1,:], linestyle='dashed', color='blue', label='HI-weighted')

ax.fill_between(ztarget, dens_cnm_w[2,0,:],  dens_cnm_w[2,2,:], facecolor='green', alpha=0.2)
ax.plot(ztarget, dens_cnm_w[2,1,:], linestyle='dashed', color='green', label='SFR-weighted')

common.prepare_legend(ax, ['red','blue', 'green'], loc = 3)
plt.tight_layout()

common.savefig(outdir, fig, 'Fractions_ISMmedia_vs_redshift.pdf')

###################################################################################################################

###################### depletion times of individual particles ####################################################
fig = plt.figure(figsize=(5,5))
ytit = "$\\tau_{\\rm particle}/\\rm Gyr$"
xtit = "redshift"
xmin, xmax, ymin, ymax = min(ztarget), max(ztarget), 0.01, 10
        

ax = fig.add_subplot(111)
common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(1,1,0.5,0.5))
ax.set_yscale('log')

ax.fill_between(ztarget, med_tauh2[0,:], med_tauh2[2,:], color='red', alpha=0.2, interpolate=True)
ax.plot(ztarget, med_tauh2[1,:], linestyle='dotted', color='red', label='H$_2$')
ax.fill_between(ztarget, med_tauhi[0,:], med_tauhi[2,:], color='blue', alpha=0.2, interpolate=True)
ax.plot(ztarget, med_tauhi[1,:], linestyle='dotted', color='blue', label='HI')

common.prepare_legend(ax, ['red','blue'], loc = 1)

plt.tight_layout()

common.savefig(outdir, fig, 'TauDep_particles_vs_redshift.pdf')

###################################################################################################################

###################### depletion times of individual particles as a function of metallicity ####################################################
fig = plt.figure(figsize=(5,5))
ytit = "$\\tau_{\\rm H_{2},particle}/\\rm Gyr$"
xtit = "$\\rm log_{10}(Z_{\\rm gas})$"
xmin, xmax, ymin, ymax = -4, -1, 0.001, 10
        

ax = fig.add_subplot(111)
common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(1,1,0.5,0.5))
ax.set_yscale('log')

ztoplot=np.array([0.0,0.5,1.0,1.5,2.0,3.5,4.0,5.0])
cols = ['Maroon', 'IndianRed', 'red', 'orange', 'gold', 'PaleGreen', 'Navy', 'Indigo']
#colors   = ('Indigo','purple','Navy','DarkTurquoise', 'Aquamarine', 'Green','PaleGreen','GreenYellow','Gold','Yellow','Orange','OrangeRed','red','DarkRed','FireBrick','Crimson','IndianRed','LightCoral','Maroon','brown','Sienna','SaddleBrown','Chocolate','Peru','DarkGoldenrod','Goldenrod','SandyBrown')

for j in range(0,len(ztoplot)):
    findz = np.where((ztarget < ztoplot[j] + 0.01) & (ztarget > ztoplot[j] - 0.01))
    print(ztarget[findz])
    if(len(ztarget[findz]) > 0):
       xplot = zbins_sfr_z[findz,:]
       yplot = tau_sfr_z[findz,:]
       yplot = yplot[0]
       xplot = xplot[0]
       ymed = yplot[0,0,:]
       ydn = yplot[0,1,:]
       yup = yplot[0,2,:]
       ax.fill_between(xplot[0,:], ydn, yup, color=cols[j], alpha=0.2, interpolate=True)
       ax.plot(xplot[0,:], ymed, linestyle='dotted', color=cols[j] , label='z=%s' % str(ztoplot[j]))

common.prepare_legend(ax, cols, loc = 2)

plt.tight_layout()

common.savefig(outdir, fig, 'TauDep_particles_vs_metallicity_vs_redshift.pdf')

###################################################################################################################

###################### depletion times of individual particles as a function of dust mass ####################################################
fig = plt.figure(figsize=(5,5))
ytit = "$\\tau_{\\rm H_{2},particle}/\\rm Gyr$"
xtit = "$\\rm log_{10}(m_{\\rm dust}/M_{\\odot})$"
xmin, xmax, ymin, ymax = 1.5, 5.5, 0.001, 10
        

ax = fig.add_subplot(111)
common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(1,1,0.5,0.5))
ax.set_yscale('log')

ztoplot=np.array([0.0,0.5,1.0,1.5,2.0,3.5,4.0,5.0])
cols = ['Maroon', 'IndianRed', 'red', 'orange', 'gold', 'PaleGreen', 'Navy', 'Indigo']
#colors   = ('Indigo','purple','Navy','DarkTurquoise', 'Aquamarine', 'Green','PaleGreen','GreenYellow','Gold','Yellow','Orange','OrangeRed','red','DarkRed','FireBrick','Crimson','IndianRed','LightCoral','Maroon','brown','Sienna','SaddleBrown','Chocolate','Peru','DarkGoldenrod','Goldenrod','SandyBrown')

for j in range(0,len(ztoplot)):
    findz = np.where((ztarget < ztoplot[j] + 0.01) & (ztarget > ztoplot[j] - 0.01))
    print(ztarget[findz])
    if(len(ztarget[findz]) > 0):
       xplot = mdbins_sfr_z[findz,:]
       yplot = tau_sfr_md[findz,:]
       yplot = yplot[0]
       xplot = xplot[0]
       ymed = yplot[0,0,:]
       ydn = yplot[0,1,:]
       yup = yplot[0,2,:]
       ax.fill_between(xplot[0,:], ydn, yup, color=cols[j], alpha=0.2, interpolate=True)
       ax.plot(xplot[0,:], ymed, linestyle='dotted', color=cols[j] , label='z=%s' % str(ztoplot[j]))

common.prepare_legend(ax, cols, loc = 2)

plt.tight_layout()

common.savefig(outdir, fig, 'TauDep_particles_vs_dustmass_vs_redshift.pdf')

###################################################################################################################


