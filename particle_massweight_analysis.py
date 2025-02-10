
import math

import numpy as np

import common
#import utilities_statistics as us


#################################################################################

################## select the model and redshift you want #######################
#model_name = 'L0100N0752/Thermal_non_equilibrium/'
#model_name = 'L0050N0752/Thermal_non_equilibrium/'
model_name = 'L0025N0376/Thermal_non_equilibrium/'
model_dir = '/cosma8/data/dp004/colibre/Runs/' + model_name

#definitions below correspond to z=0
#snap_files = ['0127', '0119', '0114', '0092', '0064', '0056', '0048', '0040', '0026', '0018']
#zstarget = [0.0, 0.1, 0.2, 1.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]

#snap_files = ['0123', '0088', '0072', '0060', '0048', '0040'] #, '0026', '0020']
#zstarget = [0.0, 1.0, 2.0, 3.5, 4.0, 5.0, 6.0] #, 8.0, 10.0]

snap_files = ['0123', '0115',  '0110',  '0106', '0102', '0098', '0096',  '0094', '0092',  '0090', '0088',  '0084', '0080', '0076', '0072', '0068', '0064', '0060', '0056', '0052', '0048', '0044', '0040', '0036', '0032', '0028']
zstarget = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]

#snap_files = ['0064', '0060', '0052', '0048', '0040']
#zstarget = [3.5, 4.0, 4.5, 5.0, 6.0]

#################################################################################
###################### simulation units #########################################
Lu = 3.086e+24/(3.086e+24) #cMpc
Mu = 1.988e+43/(1.989e+33) #Msun
tu = 3.086e+19/(3.154e+7) #yr
Tempu = 1 #K
density_cgs_conv = 6.767905773162602e-31 #conversion from simulation units to CGS for density
mH = 1.6735575e-24 #in gr
#################################################################################

def distance_3d(x,y,z, coord):
    return np.sqrt((coord[:,0]-x)**2 + (coord[:,1] - y)**2 + (coord[:,2] - z)**2)

##### loop through redshifts ######
for z in range(0,5): #len(snap_files)):
    snap_file =snap_files[z]
    ztarget = zstarget[z]
    comov_to_physical_length = 1.0 / (1.0 + ztarget)

    ################# read galaxy properties #########################################
    #fields_fof = /SOAP/HostHaloIndex, 
    #/InputHalos/HBTplus/HostFOFId
    fields_sgn = {'InputHalos': ('HaloCatalogueIndex', 'IsCentral')} 
    fields ={'ExclusiveSphere/30kpc': ('StellarMass', 'StarFormationRate', 'HalfMassRadiusStars', 'CentreOfMass', 'AtomicHydrogenMass', 'MolecularHydrogenMass', 'KappaCorotStars', 'KappaCorotGas', 'DiscToTotalStellarMassFraction', 'SpinParameter', 'MassWeightedMeanStellarAge', 'LogarithmicMassWeightedDiffuseOxygenOverHydrogenOfGasLowLimit' ,'LogarithmicMassWeightedDiffuseOxygenOverHydrogenOfGasHighLimit', 'AngularMomentumStars')}
    h5data_groups = common.read_group_data_colibre(model_dir, snap_file, fields)
    h5data_idgroups = common.read_group_data_colibre(model_dir, snap_file, fields_sgn)
    (m30, sfr30, r50, cp, mHI, mH2, kappacostar, kappacogas, disctotot, spin, stellarage, ZgasLow, ZgasHigh, Jstars) = h5data_groups

    #unit conversion
    m30 = m30 * Mu
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
    select = np.where((m30 >=3e8) & (sfr30 > 0))
    ngals = len(m30[select])
    if(ngals > 0):
       print("Number of galaxies of interest", ngals, " at redshift", ztarget, " at index", z)
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
       spin_in = spin[select]
       stellarage_in = stellarage[select]
       ZgasLow_in = ZgasLow[select]
       ZgasHigh_in = ZgasHigh[select]
       x_in = xg[select]
       y_in = yg[select]
       z_in = zg[select]
       Jstars_in = Jstars[select, :]
       spin_vec_norm = Jstars_in / np.sqrt( Jstars_in[:,0]**2 + Jstars_in[:,1]**2 + Jstars_in[:,2]**2) #normalise Jstars vector. Needed to find the plane of rotation
       spin_vec_norm = spin_vec_norm[0] #reduce dimensionality
   
       #initialise profile arrays
       mparts = np.array([])
       mdustparts = np.array([])
       mHIparts = np.array([])
       mH2parts = np.array([])
       Tparts = np.array([])
       densparts = np.array([])
       sfrparts = np.array([])
       zgasparts = np.array([])
 
       ################################# read particle data #####################################################
       fields = {'PartType0': ('GroupNr_bound', 'Coordinates' , 'Masses', 'StarFormationRates', 'Temperatures', 'SpeciesFractions', 'ElementMassFractions', 'ElementMassFractionsDiffuse', 'DustMassFractions', 'Densities')}
       h5data = common.read_particle_data_colibre(model_dir, snap_file, fields)
       
       
       #  SpeciesFractions: "elec", "HI", "HII", "Hm", "HeI", "HeII", "HeIII", "H2", "H2p", "H3p" (for snapshots)
       #  SpeciesFractions: "HI", "HII",  "H2" (snipshots to be confirmed)
       #  ElementMassFractions: "Hydrogen", "Helium", "Carbon", "Nitrogen", "Oxygen", "Neon", "Magnesium", "Silicon", "Iron", "Strontium", "Barium", "Europium"
       
       #SubhaloID is now unique to the whole box, so we only need to match a signel number
       (sgnp, coord, m, sfr, temp, speciesfrac, elementmassfracs, elementmassfracsdiff, DustMassFrac, dens) = h5data
       #get the total dust mass fraction by summing over all the dust grains
       DustMassFracTot = DustMassFrac[:,0] + DustMassFrac[:,1] + DustMassFrac[:,2] + DustMassFrac[:,3] + DustMassFrac[:,4] + DustMassFrac[:,5]
       
       #units
       coord = coord * Lu * comov_to_physical_length
       m = m * Mu
       sfr = sfr * Mu / tu
       dens = dens * density_cgs_conv / mH #in cm^-3

       ###########################################################################################################
       ############## now calculate maps of properties and save them ############################################
       
      
       #select particles that belong to the different galaxies
       #loop through galaxies
       for g in range(0,ngals):
           #select particles type 0 with the same Subhalo ID
           partin = np.where(sgnp == sgn_in[g])
           npartingal = len(sgnp[partin])
           if(npartingal > 0):
              coord_in_p0 = coord[partin,:]
              coord_in_p0 = coord_in_p0[0]
              dcentre =  distance_3d(x_in[g], y_in[g], z_in[g], coord_in_p0)
              #define vectors with mass and SFR of particles
              m_part0 = m[partin]
              sfr_part0 = sfr[partin]
              temp_part0 = temp[partin]
              dens_part0 = dens[partin]
              mdust_part0 = DustMassFracTot[partin]
              speciesfrac_part0 = speciesfrac[partin,:]
              elementmassfracs_part0 = elementmassfracs[partin,:]
              #reduce dimensionality
              elementmassfracs_part0 = elementmassfracs_part0[0]
              speciesfrac_part0 = speciesfrac_part0[0]
              inr = np.where(dcentre <= 50 * 1e-3)
              if(len(dcentre[inr]) > 0):
                    temp_inr = temp_part0[inr]
                    dens_inr = dens_part0[inr]
                    sfr_inr = sfr_part0[inr]
                    m_inr = m_part0[inr] 
                    mh_inr = m_part0[inr] * elementmassfracs_part0[inr,0]
                    mHI_inr = mh_inr * speciesfrac_part0[inr,1]
                    mH2_inr = mh_inr * speciesfrac_part0[inr,7] * 2
                    zgas_inr = m_part0[inr] * elementmassfracs_part0[inr,4] / mh_inr
                    mdust_inr = mdust_part0[inr] * m_inr
                    mparts = np.append(mparts, m_inr)
                    mHIparts = np.append(mHIparts, mHI_inr)
                    mH2parts = np.append(mH2parts, mH2_inr)
                    Tparts = np.append(Tparts, temp_inr)
                    densparts = np.append(densparts,dens_inr)
                    sfrparts = np.append(sfrparts,sfr_inr)
                    zgasparts = np.append(zgasparts,zgas_inr)
                    mdustparts = np.append(mdustparts, mdust_inr)
       print(mHIparts.shape)
       part_allprops = np.zeros(shape = (8,len(mHIparts)))
       part_allprops[0,:] = mHIparts
       part_allprops[1,:] = mH2parts
       part_allprops[2,:] = Tparts
       part_allprops[3,:] = densparts
       part_allprops[4,:] = sfrparts
       part_allprops[5,:] = zgasparts
       part_allprops[6,:] = mparts
       part_allprops[7,:] = mdustparts

       #save particle profiles
       np.savetxt(model_name + 'particles_ap50ckpc_z' + str(ztarget) + ".txt", part_allprops)

