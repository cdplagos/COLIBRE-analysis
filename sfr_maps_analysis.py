
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
model_name = 'L0100N1504/Thermal_non_equilibrium/'
#model_name = 'L0050N0752/Thermal_non_equilibrium/'
#model_name = 'L0025N0376/Thermal_non_equilibrium/'
model_dir = '/cosma8/data/dp004/colibre/Runs/' + model_name

#definitions below correspond to z=0
snap_files = ['0127', '0119', '0114', '0102', '0092', '0076', '0064', '0056', '0048', '0040', '0032', '0026', '0018']
zstarget = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]


#snap_files = ['0056', '0048', '0040', '0026', '0018']
#zstarget = [4.0, 5.0, 6.0, 8.0, 10.0]

#snap_files = ['0123', '0088', '0072', '0064', '0060', '0048', '0040'] #, '0026', '0020']
#zstarget = [0.0, 1.0, 2.0, 3.0, 3.5, 4.0, 5.0, 6.0] #, 8.0, 10.0]

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
for z in range(3,5): #,len(snap_files)):
    snap_file =snap_files[z]
    ztarget = zstarget[z]
    comov_to_physical_length = 1.0 / (1.0 + ztarget)

    ################# read galaxy properties #########################################
    #fields_fof = /SOAP/HostHaloIndex, 
    #/InputHalos/HBTplus/HostFOFId
    fields_sgn = {'InputHalos': ('HaloCatalogueIndex', 'IsCentral')} 
    fields ={'ExclusiveSphere/30kpc': ('StellarMass', 'StarFormationRate', 'HalfMassRadiusStars', 'CentreOfMass', 'AtomicHydrogenMass', 'MolecularHydrogenMass', 'KappaCorotStars', 'KappaCorotGas', 'DiscToTotalStellarMassFraction', 'SpinParameter', 'MassWeightedMeanStellarAge', 'LogarithmicMassWeightedDiffuseOxygenOverHydrogenOfGasLowLimit' ,'LogarithmicMassWeightedDiffuseOxygenOverHydrogenOfGasHighLimit', 'AngularMomentumStars', 'DustLargeGrainMass', 'DustSmallGrainMass')}
    h5data_groups = common.read_group_data_colibre(model_dir, snap_file, fields)
    h5data_idgroups = common.read_group_data_colibre(model_dir, snap_file, fields_sgn)
    (m30, sfr30, r50, cp, mHI, mH2, kappacostar, kappacogas, disctotot, spin, stellarage, ZgasLow, ZgasHigh, Jstars, mdustl, mdusts) = h5data_groups

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
       spin_in = spin[select]
       stellarage_in = stellarage[select]
       ZgasLow_in = ZgasLow[select]
       ZgasHigh_in = ZgasHigh[select]
       x_in = xg[select]
       y_in = yg[select]
       z_in = zg[select]
       Jstars_in = Jstars[select, :]
       mdust_in = mdust[select]
       spin_vec_norm = Jstars_in / np.sqrt( Jstars_in[:,0]**2 + Jstars_in[:,1]**2 + Jstars_in[:,2]**2) #normalise Jstars vector. Needed to find the plane of rotation
       spin_vec_norm = spin_vec_norm[0] #reduce dimensionality
   
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
       gal_props[:,13] = spin_in
       gal_props[:,14] = stellarage_in
       gal_props[:,15] = ZgasLow_in
       gal_props[:,16] = ZgasHigh_in
       gal_props[:,17] = mdust_in
       np.savetxt('Runs/' + model_name + '/ProcessedData/' + 'GalaxyProperties_z' + str(ztarget) + '.txt', gal_props)
       print("Have saved galaxy properties") 
       del(gal_props) #releasing memory
 
       #initialise profile arrays
       sfr_profile = np.zeros(shape = (ngals, nr))
       mHI_profile = np.zeros(shape = (ngals, nr))
       mH2_profile = np.zeros(shape = (ngals, nr))
       mstar_profile = np.zeros(shape = (ngals, nr))
       oh_profile = np.zeros(shape = (ngals, nr))
       feh_profile = np.zeros(shape = (ngals, nr))
       coldgas_profile = np.zeros(shape = (ngals, nr))
       mdust_profile = np.zeros(shape = (ngals, nr))
       npart0_profile = np.zeros(shape = (ngals, nr))
       npart4_profile = np.zeros(shape = (ngals, nr))
       r_dist_centre = np.zeros(shape = nr) 
       if (family_method == 'radial_profiles'):
            r_dist_centre = xr
       elif (family_method == 'grid'):
            for i,ri in enumerate(gr):
                for j,rj in enumerate(gr):
                    r_dist_centre[i * len(gr) + j] = np.sqrt(ri**2 + rj**2)
 
       ################################# read particle data #####################################################
       fields = {'PartType0': ('GroupNr_bound', 'Coordinates' , 'Masses', 'StarFormationRates', 'Temperatures', 'SpeciesFractions', 'ElementMassFractions', 'ElementMassFractionsDiffuse', 'DustMassFractions', 'Densities')}
       print("Will read part type 0 properties")
       h5data = common.read_particle_data_colibre(model_dir, snap_file, fields)
       print("Have read particle 0 properties")
       
       #  SpeciesFractions: "elec", "HI", "HII", "Hm", "HeI", "HeII", "HeIII", "H2", "H2p", "H3p" (for snapshots)
       #  SpeciesFractions: "HI", "HII",  "H2" (snipshots to be confirmed)
       #  ElementMassFractions: "Hydrogen", "Helium", "Carbon", "Nitrogen", "Oxygen", "Neon", "Magnesium", "Silicon", "Iron", "Strontium", "Barium", "Europium"
       
       #SubhaloID is now unique to the whole box, so we only need to match a signel number
       (sgnp, coord, m, sfr, temp, speciesfrac, elementmassfracs, elementmassfracsdiff, DustMassFrac, dens) = h5data
      
       del(h5data) #release data

       speciesfrac2 = np.zeros(shape = (len(speciesfrac[:,0]), 2))
       speciesfrac2[:,0] = speciesfrac[:,1]
       speciesfrac2[:,1] = speciesfrac[:,7]
       speciesfrac = speciesfrac2
       del(speciesfrac2)

       def reduce_dim_elements(elementmassfracs):
           els = np.zeros(shape = (len(elementmassfracs[:,0]), 3))
           els[:,0] = elementmassfracs[:,0]
           els[:,1] = elementmassfracs[:,4]
           els[:,2] = elementmassfracs[:,8]
           elementmassfracs = els
           del(els)

       reduce_dim_elements(elementmassfracs)
       reduce_dim_elements(elementmassfracsdiff)

       #fields_in = np.isin(sgnp,sgn_in) #find all particles of interest and redefine arrays with only those
       #sgnp = sgnp[fields_in]
       #coord = coord[fields_in]
       #m = m[fields_in]
       #sfr = sfr[fields_in]
       #temp = temp[fields_in]
       #speciesfrac = speciesfrac[fields_in,:]
       #elementmassfracs = elementmassfracs[fields_in,:]
       #elementmassfracsdiff = elementmassfracsdiff[fields_in,:]
       #DustMassFrac = DustMassFrac[fields_in,:]
       #dens = dens[fields_in]
       #del(fields_in) #redefine array to release memory

       #get the total dust mass fraction by summing over all the dust grains
       DustMassFracTot = DustMassFrac[:,0] + DustMassFrac[:,1] + DustMassFrac[:,2] + DustMassFrac[:,3] + DustMassFrac[:,4] + DustMassFrac[:,5]
       
       np.delete(DustMassFrac)
       print("Will read part type 4 properties")
       fields = {'PartType4': ('GroupNr_bound', 'Coordinates' , 'Masses')}
       h5data = common.read_particle_data_colibre(model_dir, snap_file, fields)
       print("Have read particle 4 properties")
       #SubhaloID is now unique to the whole box, so we only need to match a single number
       (sgnpT4, coordT4, mT4) = h5data
       del(h5data) #release data
       #fields_in = np.isin(sgnpT4,sgn_in) #find all particles of interest and redefine arrays with only those
        
       #sgnpT4 = sgnpT4[fields_in]
       #coordT4 = coordT4[fields_in]
       #mT4 = mT4[fields_in]
       #del(fields_in) #redefine array to release memory

       #units
       coord = coord * Lu * comov_to_physical_length
       m = m * Mu
       sfr = sfr * Mu / tu
       coordT4 = coordT4 * Lu * comov_to_physical_length
       mT4 = mT4 * Mu
       dens = dens * density_cgs_conv / mH #in cm^-3
       ###########################################################################################################
       ############## now calculate maps of properties and save them ############################################
       
      
       #select particles that belong to the different galaxies
       #loop through galaxies
       print("Will compute radial profiles")
       for g in range(0,ngals):
           #select particles type 0 with the same Subhalo ID
           partin = np.where(sgnp == sgn_in[g])
           npartingal = len(sgnp[partin])
           if(npartingal > 0):
              coord_in_p0 = coord[partin,:]
              coord_in_p0 = coord_in_p0[0]
              if (family_method == 'radial_profiles'):
                  #calculate distance between particle and centre of potential
                  if (method == 'spherical_apertures'):
                      dcentre =  distance_3d(x_in[g], y_in[g], z_in[g], coord_in_p0)
                  elif (method == 'circular_apertures_random_map'):
                      dcentre = distance_2d_random(x_in[g], y_in[g], coord_in_p0)
                  elif (method == 'circular_apertures_face_on_map'):
                      dcentre = distance_2d_faceon(x_in[g], y_in[g], z_in[g], coord_in_p0, spin_vec_norm[g,:])
              #define vectors with mass and SFR of particles
              m_part0 = m[partin]
              sfr_part0 = sfr[partin]
              temp_part0 = temp[partin]
              dust_part0 = DustMassFracTot[partin]
              elementmassfracs_part0 = elementmassfracs[partin,:]
              elementmassfracsdiff_part0 = elementmassfracsdiff[partin,:]
              dens_part0 = dens[partin]
              speciesfrac_part0 = speciesfrac[partin,:]
              #reduce dimensionality
              elementmassfracs_part0 = elementmassfracs_part0[0]
              elementmassfracsdiff_part0 = elementmassfracsdiff_part0[0]
              speciesfrac_part0 = speciesfrac_part0[0]
              if (family_method == 'radial_profiles'):
                  for i,r in enumerate(xr):
                      inr = np.where((dcentre >= (r - dr/2)*1e-3) & (dcentre < (r + dr/2)*1e-3))
                      if(len(dcentre[inr]) > 0):
                         npart0_profile[g,i] = len(dcentre[inr])
                         temp_inr = temp_part0[inr]
                         dens_inr = dens_part0[inr]
                         mh_inr = m_part0[inr] * elementmassfracs_part0[inr,0]
                         mhdiff_inr = m_part0[inr] * elementmassfracsdiff_part0[inr,0]
                         mo_inr = m_part0[inr] * elementmassfracsdiff_part0[inr,1]
                         mfe_inr = m_part0[inr] * elementmassfracsdiff_part0[inr,2]
                         sfr_inr = sfr_part0[inr]
                         #reduce dimensionality
                         mh_inr = mh_inr[0]
                         mhdiff_inr = mhdiff_inr[0]
                         mo_inr = mo_inr[0]
                         mfe_inr = mfe_inr[0]
                         mpart_inr = m_part0[inr]
                         #calculate profiles
                         sfr_profile[g,i] = np.sum(sfr_inr)
                         mHI_profile[g,i] = np.sum(mh_inr * speciesfrac_part0[inr,0])
                         mH2_profile[g,i] = np.sum(mh_inr * speciesfrac_part0[inr,1] * 2) #factor 2 comes from H2 being two hydrogen atoms
                         mdust_profile[g,i] = np.sum(mpart_inr * dust_part0[inr])
       
                         coldp = np.where((temp_inr < 10**(4.5)) & (dens_inr > 0)) #select particles with temperatures cooler than 10^4.5K and calculate metallicity profiles with those particles only.
                         if(len(mo_inr[coldp]) > 0):
                            oh_profile[g,i] = np.sum(mo_inr[coldp]) / np.sum(mhdiff_inr[coldp])
                            feh_profile[g,i] = np.sum(mfe_inr[coldp]) / np.sum(mhdiff_inr[coldp])
                            coldgas_profile[g,i] = np.sum(mpart_inr[coldp])
                         del(temp_inr, dens_inr, mh_inr, mhdiff_inr, mo_inr, mfe_inr, sfr_inr, coldp) #release data

              elif (family_method == 'grid'):
                  if(method == 'grid_random_map'):
                      dcentre_i, dcentre_j = distance_2d_grid_random(x_in[g], y_in[g], coord_in_p0)
                  for i,ri in enumerate(gr):
                      inr = np.where((dcentre_i >= (ri - dg/2)*1e-3) & (dcentre_i < (ri + dg/2)*1e-3))
                      if(len(dcentre_i[inr]) > 0):
                          temp_inr = temp_part0[inr]
                          dens_inr = dens_part0[inr]
                          mh_inr = m_part0[inr] * elementmassfracs_part0[inr,0]
                          mhdiff_inr = m_part0[inr] * elementmassfracsdiff_part0[inr,0]
                          mo_inr = m_part0[inr] * elementmassfracsdiff_part0[inr,1]
                          mfe_inr = m_part0[inr] * elementmassfracsdiff_part0[inr,2]
                          speciesfrac_inr = speciesfrac_part0[inr,:]
                          dust_part_inr = dust_part0[inr]
                          #reduce dimensionality
                          mh_inr = mh_inr[0]
                          mhdiff_inr = mhdiff_inr[0]
                          mo_inr = mo_inr[0]
                          mfe_inr = mfe_inr[0]
                          mpart_inr = m_part0[inr]
                          speciesfrac_inr = speciesfrac_inr[0]

                          sfr_inr = sfr_part0[inr]
                          dcentre_j_inr = dcentre_j[inr]
                          for j,rj in enumerate(gr):
                              inrj = np.where((dcentre_j_inr >= (rj - dg/2)*1e-3) & (dcentre_j_inr < (rj + dg/2)*1e-3))
                              if(len(dcentre_j_inr[inrj]) > 0):
                                  #calculate profiles
                                  npart0_profile[g,i * len(gr) + j] = len(dcentre_j_inr[inrj])
                                  sfr_profile[g,i * len(gr) + j] = np.sum(sfr_inr[inrj])
                                  mHI_profile[g,i * len(gr) + j] = np.sum(mh_inr[inrj] * speciesfrac_inr[inrj,0])
                                  mH2_profile[g,i * len(gr) + j] = np.sum(mh_inr[inrj] * speciesfrac_inr[inrj,1] * 2) #factor 2 comes from H2 being two hydrogen atoms
                                  mdust_profile[g,i * len(gr) + j] = np.sum(mpart_inr[inrj] * dust_part_inr[inrj])
                                  temp_inj = temp_inr[inrj]
                                  dens_inj = dens_inr[inrj]
                                  mh_inj = mhdiff_inr[inrj]
                                  mo_inj = mo_inr[inrj] 
                                  mfe_inj = mfe_inr[inrj] 
                                  mpart_inj = mpart_inr[inrj]
                                  coldp = np.where((temp_inj < 10**(4.5)) & (dens_inj > 0)) #select particles with temperatures cooler than 10^4.5K and calculate metallicity profiles with those particles only.
                                  if(len(mo_inj[coldp]) > 0):
                                     oh_profile[g,i * len(gr) + j] = np.sum(mo_inj[coldp]) / np.sum(mh_inj[coldp])
                                     feh_profile[g,i * len(gr) + j] = np.sum(mfe_inj[coldp]) / np.sum(mh_inj[coldp])
                                     coldgas_profile[g,i * len(gr) + j] = np.sum(mpart_inj[coldp])
                                  del(temp_inj,dens_inj,mh_inj,mo_inj,mfe_inj,mpart_inj,coldp)
                          del(dcentre_i, dcentre_j,temp_inr,dens_inr,mh_inr,mhdiff_inr,mo_inr,mfe_inr,speciesfrac_inr,dust_part_inr,sfr_inr,dcentre_j_inr)        
              del(m_part0, sfr_part0, temp_part0, dust_part0, elementmassfracs_part0, elementmassfracsdiff_part0, dens_part0, speciesfrac_part0) #release data
      
           #select particles type 4 with the same Subhalo ID
           partin = np.where(sgnpT4 == sgn_in[g])
           npartingal = len(sgnpT4[partin])
           if(npartingal > 0):
              coord_in_p4 = coordT4[partin,:]
              coord_in_p4 = coord_in_p4[0]
              if (family_method == 'radial_profiles'):
                   #calculate distance between particle and centre of potential
                   if (method == 'spherical_apertures'):
                       dcentre =  distance_3d(x_in[g], y_in[g], z_in[g], coord_in_p4)
                   elif (method == 'circular_apertures_random_map'):
                       dcentre = distance_2d_random(x_in[g], y_in[g], coord_in_p4)
                   elif (method == 'circular_apertures_face_on_map'):
                       dcentre = distance_2d_faceon(x_in[g], y_in[g], z_in[g], coord_in_p4, spin_vec_norm[g,:])
              #define vectors with mass and SFR of particles
              m_part4 = mT4[partin]
              if (family_method == 'radial_profiles'):
                  for i,r in enumerate(xr):
                      inr = np.where((dcentre >= (r - dr/2)*1e-3) & (dcentre < (r + dr/2)*1e-3) & (m_part4 > 0))
                      if(len(dcentre[inr]) > 0):
                          npart4_profile[g,i] = len(dcentre[inr])
                          mstar_profile[g,i] = np.sum(m_part4[inr])
                  del(dcentre,inr) 
              elif (family_method == 'grid'):
                 if(method == 'grid_random_map'):
                     dcentre_i, dcentre_j = distance_2d_grid_random(x_in[g], y_in[g], coord_in_p4)
                 for i,ri in enumerate(gr):
                     inr = np.where((dcentre_i >= (ri - dg/2)*1e-3) & (dcentre_i < (ri + dg/2)*1e-3))
                     if(len(dcentre_i[inr]) > 0):
                         mp4_inr = m_part4[inr] 
                         dcentre_j_inr = dcentre_j[inr]
                         for j,rj in enumerate(gr):
                             inrj = np.where((dcentre_j_inr >= (rj - dg/2)*1e-3) & (dcentre_j_inr < (rj + dg/2)*1e-3))
                             if(len(dcentre_j_inr[inrj]) > 0):
                                 #calculate profiles
                                 npart4_profile[g,i * len(gr) + j] = len(dcentre_j_inr[inrj])
                                 mstar_profile[g,i * len(gr) + j] = np.sum(mp4_inr[inrj])
                         del(dcentre_i, dcentre_j,mp4_inr,inr,dcentre_j_inr)
              del(coord_in_p4,m_part4)

  
       #save galaxy profiles
       np.savetxt('Runs/' + model_name + '/ProcessedData/' + 'SFR_profiles_ap50ckpc_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + ".txt", sfr_profile)
       np.savetxt('Runs/' + model_name + '/ProcessedData/' +  'MHI_profiles_ap50ckpc_'+ method + "_dr"+ str(dr) + "_z"  + str(ztarget) + ".txt", mHI_profile)
       np.savetxt('Runs/' + model_name + '/ProcessedData/' +  'MH2_profiles_ap50ckpc_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + ".txt", mH2_profile)
       np.savetxt('Runs/' + model_name + '/ProcessedData/' +  'OH_gas_profiles_ap50ckpc_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + ".txt", oh_profile)
       np.savetxt('Runs/' + model_name + '/ProcessedData/' +  'FeH_gas_profiles_ap50ckpc_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + ".txt", feh_profile)
       np.savetxt('Runs/' + model_name + '/ProcessedData/' +  'Mstar_profiles_ap50ckpc_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + ".txt", mstar_profile)
       np.savetxt('Runs/' + model_name + '/ProcessedData/' +  'Mdust_profiles_ap50ckpc_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + ".txt", mdust_profile)
       np.savetxt('Runs/' + model_name + '/ProcessedData/' +  'NumberPart0_profiles_ap50ckpc_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + ".txt", npart0_profile)
       np.savetxt('Runs/' + model_name + '/ProcessedData/' +  'NumberPart4_profiles_ap50ckpc_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + ".txt", npart4_profile)
       np.savetxt('Runs/' + model_name + '/ProcessedData/' +  'Mcoldgas_profiles_ap50ckpc_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + ".txt", coldgas_profile)

       #save radii info
       np.savetxt('Runs/' + model_name + '/ProcessedData/' +  'radii_info_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + ".txt", r_dist_centre)
   
