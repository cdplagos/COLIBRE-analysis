
import math

import numpy as np

import common
#import utilities_statistics as us


###### choose what kind of profiles you want (choose one only) ##################
family_method = 'radial_profiles'
method = 'circular_apertures_face_on_map'
#method = 'circular_apertures_random_map'
#method = 'spherical_apertures'

#family_method = 'grid'
#method = 'grid_face_on_map' #to be coded
#method = 'grid_random_map'
#method = 'voronoi_maps'  #to be coded

#################################################################################

################## select the model and redshift you want #######################
#model_name = 'L0100N0752/Thermal_non_equilibrium/'
#model_name = 'L0050N0752/Thermal_non_equilibrium/'
#model_name = 'L0025N0376/Thermal/'
model_name = 'L200_m6/Thermal/'

model_dir = '/cosma8/data/dp004/colibre/Runs/' + model_name

#definitions below correspond to z=0
#snap_files = ['0127', '0119', '0114', '0102', '0092', '0076', '0064', '0056', '0048', '0040', '0026', '0018']
#zstarget = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]

snap_files = ['0102', '0092', '0076', '0064', '0056', '0048', '0040', '0032', '0026', '0018']
zstarget = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]

#snap_files = ['0056', '0048', '0040', '0026', '0018']
#zstarget = [4.0, 5.0, 6.0, 8.0, 10.0]

#snap_files = ['0123', '0088', '0072', '0060', '0048', '0040'] #, '0026', '0020']
#zstarget = [0.0, 1.0, 2.0, 3.5, 4.0, 5.0, 6.0] #, 8.0, 10.0]

#################################################################################
###################### simulation units #########################################
Lu = 3.086e+24/(3.086e+24) #cMpc
Mu = 1.988e+43/(1.989e+33) #Msun
tu = 3.086e+19/(3.154e+7) #yr
Tempu = 1 #K
density_cgs_conv = 6.767905773162602e-31 #conversion from simulation units to CGS for density
mH = 1.6735575e-24 #in gr
#################################################################################

#define radial bins of interest. This going from 0 to 50kpc, in bins of 1kpc
rmax = 50
rmin = 0
dr = 1.0
rbins = np.arange(rmin, rmax, dr)
xr = rbins + dr/2.0 
nr = len(xr) #number of radial bins

gmax = 50
gmin = -50
dg = 1.0
gbins = np.arange(gmin, gmax, dg)
gr = gbins + dg/2.0 
ng = len(gr) #number of radial bins

if(family_method == 'grid'):
    nr  = ng * ng
  
def distance_3d(x,y,z, coord):
    return np.sqrt((coord[:,0]-x)**2 + (coord[:,1] - y)**2 + (coord[:,2] - z)**2)

def distance_2d_random(x,y, coord):
    return np.sqrt((coord[:,0]-x)**2 + (coord[:,1] - y)**2)

def distance_2d_grid_random(x,y, coord):
    return (coord[:,0]-x), (coord[:,1] - y)


def distance_2d_faceon(x,y,z, coord, spin_vec):
    cross_prod = np.zeros(shape = (len(coord[:,0]), 3))
    coord_norm = np.zeros(shape = (len(coord[:,0]), 3))

    #normalise coordinate vector of particles
    normal_coord =  np.sqrt(coord[:,0]**2+ coord[:,1]**2 + coord[:,2]**2)
    coord_norm[:,0] = coord[:,0]/normal_coord
    coord_norm[:,1] = coord[:,1]/normal_coord
    coord_norm[:,2] = coord[:,2]/normal_coord

    #calculate cross product vector
    cross_prod[:,0] = (coord_norm[:,1] * spin_vec[2] - coord_norm[:,2] * spin_vec[1])
    cross_prod[:,1] = (coord_norm[:,2] * spin_vec[0] - coord_norm[:,0] * spin_vec[2])
    cross_prod[:,2] = (coord_norm[:,0] * spin_vec[1] - coord_norm[:,1] * spin_vec[0])
    #calculate angle between vectors
    sin_thetha = np.sin(np.acos(np.sqrt(cross_prod[:,0]**2 + cross_prod[:,1]**2 + cross_prod[:,2]**2)))
    #return projected distance
    dcentre3d = distance_3d(x,y,z, coord)
    return dcentre3d * sin_thetha

 
##### loop through redshifts ######
for z in range(7,8): #0,len(snap_files)):
    snap_file =snap_files[z]
    ztarget = zstarget[z]
    comov_to_physical_length = 1.0 / (1.0 + ztarget)

    ################# read galaxy properties #########################################
    #fields_fof = /SOAP/HostHaloIndex, 
    #/InputHalos/HBTplus/HostFOFId
    fields_sgn = {'InputHalos': ('HaloCatalogueIndex', 'IsCentral', 'HBTplus/DescendantTrackId', 'HBTplus/TrackId')} 
    fields ={'ExclusiveSphere/50kpc': ('StellarMass', 'StarFormationRate', 'HalfMassRadiusStars', 'CentreOfMass', 'AtomicHydrogenMass', 'MolecularHydrogenMass', 'KappaCorotStars', 'KappaCorotGas', 'DiscToTotalStellarMassFraction', 'MassWeightedMeanStellarAge', 'LogarithmicMassWeightedDiffuseOxygenOverHydrogenOfGasLowLimit' ,'LogarithmicMassWeightedDiffuseOxygenOverHydrogenOfGasHighLimit', 'AngularMomentumStars', 'DustLargeGrainMass', 'DustSmallGrainMass')}
    h5data_groups = common.read_group_data_colibre(model_dir, snap_file, fields)
    h5data_idgroups = common.read_group_data_colibre(model_dir, snap_file, fields_sgn)
    (m30, sfr30, r50, cp, mHI, mH2, kappacostar, kappacogas, disctotot, stellarage, ZgasLow, ZgasHigh, Jstars, mdustl, mdusts) = h5data_groups

    #unit conversion
    mdust = (mdustl + mdusts) * Mu
    m30 = m30 * Mu
    mHI = mHI * Mu
    mH2 = mH2 * Mu
    sfr30 = sfr30 * Mu / tu 
    r50 = r50 * Lu * comov_to_physical_length
    stellarage = stellarage * tu / 1e9 #in Gyr
    cp = cp * Lu * comov_to_physical_length
    Jstars = Jstars * Mu / (Lu * comov_to_physical_length)**2 / tu
    
    (sgn, is_central, desc_id, track_id) = h5data_idgroups
    xg = cp[:,0]
    yg = cp[:,1]
    zg = cp[:,2]
    ###################################################################################


    ######################### select galaxies of interest #############################
    select = np.where((m30 >=1e9) & (sfr30 > 0))
    ngals = len(m30[select])
    if(ngals > 0):
       print("Number of galaxies of interest", ngals, " at redshift", ztarget)
       m_in = m30[select]
       sfr_in = sfr30[select]
       sgn_in = sgn[select]
       is_central_in = is_central[select]
       r50_in = r50[select]
       mHI_in = mHI[select]
       mH2_in = mH2[select]
       kappacostar_in = kappacostar[select]
       kappacogas_in = kappacogas[select]
       disctotot_in = disctotot[select]
       stellarage_in = stellarage[select]
       ZgasLow_in = ZgasLow[select]
       ZgasHigh_in = ZgasHigh[select]
       x_in = xg[select]
       y_in = yg[select]
       z_in = zg[select]
       Jstars_in = Jstars[select, :]
       Jstars_in = Jstars_in[0]
       Jstars_in_norm = np.sqrt(Jstars_in[:,0]**2 + Jstars_in[:,1]**2 + Jstars_in[:,2]**2)
       mdust_in = mdust[select]
   
       #save galaxy properties of interest
       gal_props = np.zeros(shape = (ngals, 20))
       gal_props[:,0] = sgn_in
       gal_props[:,1] = is_central_in
       gal_props[:,2] = x_in
       gal_props[:,3] = y_in
       gal_props[:,4] = z_in
       gal_props[:,5] = m_in
       gal_props[:,6] = sfr_in
       gal_props[:,7] = r50_in
       gal_props[:,8] = mHI_in
       gal_props[:,9] = mH2_in
       gal_props[:,10] = kappacostar_in
       gal_props[:,11] = kappacogas_in
       gal_props[:,12] = disctotot_in
       gal_props[:,13] = Jstars_in_norm
       gal_props[:,14] = stellarage_in
       gal_props[:,15] = ZgasLow_in
       gal_props[:,16] = ZgasHigh_in
       gal_props[:,17] = mdust_in
       gal_props[:,18] = desc_id[select]
       gal_props[:,19] = track_id[select]

       np.savetxt('Runs/' + model_name + '/ProcessedData/GalaxyProperties_z' + str(ztarget) + '.txt', gal_props)
       
   
