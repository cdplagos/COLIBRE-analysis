
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import common
import utilities_statistics as us

import argparse

parser = argparse.ArgumentParser(description="Takes two arguments.")

parser.add_argument('snap', type=str)
parser.add_argument('redshift', type=float)
parser.add_argument('subv', type=int)
# Parse the arguments
args = parser.parse_args()

###### choose what kind of profiles you want (choose one only) ##################
family_method = 'radial_profiles'
method = 'circular_apertures_face_on_map'
#method = 'circular_apertures_random_map'
#method = 'spherical_apertures'

#family_method = 'grid'
#method = 'grid_face_on_map' 
#method = 'grid_random_map'
#method = 'voronoi_maps'  #to be coded

#################################################################################

################## select the model and redshift you want #######################
#model_name = 'L0100N1504/Thermal/'
#model_name = 'L0050N0752/Thermal/'
#model_name = 'L0025N0376/Thermal/'
#model_name = 'L0025N0188/Thermal/'
#model_name = 'L0025N0752/Thermal/'
model_name = 'L200_m6/Thermal/'
#model_name = 'L0050N0752/HYBRID_AGN_m6/'
model_dir = '/cosma8/data/dp004/colibre/Runs/' + model_name
out_dir = '/cosma8/data/dp004/ngdg66/Runs/' + model_name + '/'

sm_limit = 1e9

#definitions below correspond to z=0
#snap_files = ['0127', '0119', '0114', '0102', '0092', '0076', '0064', '0056', '0048', '0040', '0032', '0026', '0018']
#zstarget = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
#snap_type = [True,True,True,True,True,True,True,True,True,True,True,True,True]
#snap_files = ['0127', '0102', '0092', '0076', '0064', '0056', '0048', '0040', '0032', '0026', '0018']
#zstarget = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
##snap_type = [True, True, True,True,True, True,True,True,True,True,True,True]
#snap_files = ['0122', '0110', '0106','0100','0098','0096','0094','0088','0084','0080','0072','0068']
#zstarget = [0.05,0.3,0.4,0.6,0.7,0.8,0.9,1.125,1.5,1.75,2.25,2.5]
#snap_type = [True, True, True,True,True, True,True,True,True,True,True,True]

#snap_files = ['0100','0096','0088','0084','0080','0072','0068']
#zstarget = [0.6,0.8,1.125,1.5,1.75,2.25,2.5]
#snap_type = [True,True,True,True,True,True,True,True]


#snap_files = ['0040', '0026', '0018']
#zstarget = [6.0, 8.0, 10.0]
#snap_type = [True,True,True]

#snap_files = ['0123', '0088', '0072', '0064', '0060', '0048', '0040'] #, '0026', '0020']
#zstarget = [0.0, 1.0, 2.0, 3.0, 3.5, 4.0, 5.0, 6.0] #, 8.0, 10.0]
#snap_type = [True,True,True,True,True,True,True,True]


#snap_files = ['0102', '0092', '0076', '0064', '0056', '0048', '0040', '0026', '0018']
#zstarget = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
#snap_type = [True,True,True,True,True,True,True,True,True]
#################################################################################
###################### simulation units #########################################
Lu = 3.086e+24/(3.086e+24) #cMpc
Mu = 1.988e+43/(1.989e+33) #Msun
tu = 3.086e+19/(3.154e+7) #yr
Tempu = 1 #K
density_cgs_conv = 6.767905773162602e-31 #conversion from simulation units to CGS for density
velocity_cgs_conv = 9.785e+11 #from Mpc/yr to km/s
mH = 1.6735575e-24 #in gr
#################################################################################
def distance_3d(x,y,z, coord):
    return np.sqrt((coord[:,0]-x)**2 + (coord[:,1] - y)**2 + (coord[:,2] - z)**2)

#define time bins of interest. This going from 0 to 14.5Gyr, in bins of 10Myr

gmax = 14.5
gmin = 0
dg = 0.01
gbins = np.arange(gmin, gmax, dg)
gr = gbins + dg/2.0 
ng = len(gr) #number of time bins

##### loop through redshifts ######
snap_file = args.snap
ztarget = args.redshift
subvolume = args.subv
comov_to_physical_length = 1.0 / (1.0 + ztarget)

################# read galaxy properties #########################################
#fields_fof = /SOAP/HostHaloIndex, 
#/InputHalos/HBTplus/HostFOFId
fields_sgn = {'InputHalos': ('HaloCatalogueIndex', 'IsCentral')} 
fields ={'ExclusiveSphere/50kpc': ('StellarMass', 'StarFormationRate', 'HalfMassRadiusStars', 'AtomicHydrogenMass', 'MolecularHydrogenMass', 'KappaCorotStars', 'KappaCorotGas', 'DiscToTotalStellarMassFraction', 'MassWeightedMeanStellarAge', 'LogarithmicMassWeightedDiffuseOxygenOverHydrogenOfGasLowLimit' ,'LogarithmicMassWeightedDiffuseOxygenOverHydrogenOfGasHighLimit', 'AngularMomentumStars', 'DustLargeGrainMass', 'DustSmallGrainMass', 'CentreOfMass', 'MostMassiveBlackHoleVelocity')}
fields_cen ={'BoundSubhalo' : ('CentreOfMass','MostMassiveBlackHoleVelocity')}
h5data_groups = common.read_group_data_colibre(model_dir, snap_file, fields)
h5data_cen = common.read_group_data_colibre(model_dir, snap_file, fields_cen)
h5data_idgroups = common.read_group_data_colibre(model_dir, snap_file, fields_sgn)
(m30, sfr30, r50, mHI, mH2, kappacostar, kappacogas, disctotot, stellarage, ZgasLow, ZgasHigh, Jstars, mdustl, mdusts, cp, v_at_cp) = h5data_groups
#(cp, v_at_cp) = h5data_cen

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

(sgn, is_central) = h5data_idgroups
xg = cp[:,0]
yg = cp[:,1]
zg = cp[:,2]
###################################################################################



######################### select galaxies of interest #############################
select = np.where((m30 >= sm_limit) & (sfr30 >= 0))
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
   mdust_in = mdust[select]
   v_at_cp = v_at_cp[select,:]
   v_at_cp = v_at_cp[0]
   Jstars_in = Jstars_in[0]
   Jstars_in_norm = np.sqrt(Jstars_in[:,0]**2 + Jstars_in[:,1]**2 + Jstars_in[:,2]**2)
   spin_vec_norm = np.zeros(shape= (ngals,3)) 
   spin_vec_norm[:,0] = Jstars_in[:,0] / Jstars_in_norm #normalise Jstars vector. Needed to find the plane of rotation
   spin_vec_norm[:,1] = Jstars_in[:,1] / Jstars_in_norm #normalise Jstars vector. Needed to find the plane of rotation
   spin_vec_norm[:,2] = Jstars_in[:,2] / Jstars_in_norm #normalise Jstars vector. Needed to find the plane of rotation

   if(subvolume == 0):
      #save galaxy properties of interest
      gal_props = np.zeros(shape = (ngals, 18))
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
      gal_props[:,13] = Jstars_in_norm #Jstars norm
      gal_props[:,14] = stellarage_in
      gal_props[:,15] = ZgasLow_in
      gal_props[:,16] = ZgasHigh_in
      gal_props[:,17] = mdust_in
      np.savetxt(out_dir + '/ProcessedData/' + 'GalaxyProperties_sfrGE0_z' + str(ztarget) + '.txt', gal_props)
      print("Have saved galaxy properties") 
      del(gal_props) #releasing memory

   #initialise profile arrays
   mstar_sfh = np.zeros(shape = (ngals, len(gr)))
   galaxies_in_subv = np.zeros(shape = ngals)

   ################################# read particle data #####################################################

   print("Will read part type 4 properties")
   fields_gn = {'PartType4': ('GroupNr_bound', 'Rank_bound')}
   fields = {'PartType4': ('Coordinates', 'Masses', 'Velocities', 'InitialMasses', 'BirthScaleFactors')}
   h5data = common.read_particle_colibre_single_subv(model_dir, snap_file, subvolume, fields)
   print("Have read particle 4 properties")
   #SubhaloID is now unique to the whole box, so we only need to match a single number
   (coordT4, mT4, vT4, miT4, abirth) = h5data
   del(h5data) #release data
   h5data = common.read_particle_membership_colibre_single_subv(model_dir, snap_file, subvolume, fields_gn)
   (sgnpT4, _) = h5data
   del(h5data) #release data
   #fields_in = np.isin(sgnpT4,sgn_in) #find all particles of interest and redefine arrays with only those
   

   #units
   coordT4 = coordT4 * Lu * comov_to_physical_length
   mT4 = mT4 * Mu
   miT4 = miT4 * Mu
   lbt_birth = us.look_back_time(1.0/abirth - 1, h=0.681, omegam=0.306, omegal=0.693922)
   ###########################################################################################################
   ############## now calculate maps of properties and save them ############################################
   
  
   #select particles that belong to the different galaxies
   #loop through galaxies
   print("Will compute star formation histories")
   for g in range(0,ngals):

       #select particles type 4 with the same Subhalo ID
       partin4 = np.where(sgnpT4 == sgn_in[g])
       npartingalT4 = len(sgnpT4[partin4])
       if(npartingalT4 > 0):
          galaxies_in_subv[g] = sgn_in[g]
          coord_in_p4 = coordT4[partin4,:]
          coord_in_p4 = coord_in_p4[0]
          m_part4 = mT4[partin4]
          mi_part4 = miT4[partin4]
          lbt_part4 = lbt_birth[partin4]
          #compute mean velocity with all stellar particles within 1kpc.
          dcentre_3d_T4 =  distance_3d(x_in[g], y_in[g], z_in[g], coord_in_p4)
          in50kpc = np.where(dcentre_3d_T4 < 50.0 *1e-3)
          if(len(m_part4[in50kpc]) > 0):
             mi_part4_in = mi_part4[in50kpc]
             lbt_part4_in = lbt_part4[in50kpc]

             for i,ti in enumerate(gr):
                 inb = np.where((lbt_part4_in >= (ti - dg/2)) & (lbt_part4_in < (ti + dg/2)))
                 if(len(lbt_part4_in[inb]) > 0):
                     mstar_sfh[g,i] = np.sum(mi_part4_in[inb])
   

   #select galaxies that were processed:
   inw = np.where(galaxies_in_subv != 0)
   print("Number of galaxies processed in this subvolume", len(galaxies_in_subv[inw]), " out of", ngals) 
   msin  = mstar_sfh[inw,:] 
   wedge_name = 'subvolume_' + str(subvolume)
   np.savetxt(out_dir + '/ProcessedData/' + 'Galaxies_sfrGE0_in_subv_z' + str(ztarget) + wedge_name + ".txt", galaxies_in_subv[inw])
   np.savetxt(out_dir + '/ProcessedData/' +  'Mstar_SFH_ap50ckpc_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + wedge_name + ".txt", msin[0])

   if(subvolume == 0): 
      #save radii info
      np.savetxt(out_dir + '/ProcessedData/' +  'look_back_time_info_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + ".txt", gr)
   
