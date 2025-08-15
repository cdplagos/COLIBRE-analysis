
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import common
#import utilities_statistics as us


###### choose what kind of profiles you want (choose one only) ##################
#family_method = 'radial_profiles'
#method = 'circular_apertures_face_on_map'
#method = 'circular_apertures_random_map'
#method = 'spherical_apertures'

family_method = 'grid'
method = 'grid_face_on_map' 
#method = 'grid_random_map'
#method = 'voronoi_maps'  #to be coded

#################################################################################

################## select the model and redshift you want #######################
#model_name = 'L0100N1504/Thermal/'
#model_name = 'L0050N0752/Thermal/'
model_name = 'L0025N0376/Thermal/'
#model_name = 'L0025N0188/Thermal/'
#model_name = 'L0025N0752/Thermal/'
#model_name = 'L200_m6/Thermal/'

model_dir = '/cosma8/data/dp004/colibre/Runs/' + model_name

sm_limit = 1e9

#definitions below correspond to z=0
#snap_files = ['0127', '0119', '0114', '0102', '0092', '0076', '0064', '0056', '0048', '0040', '0032', '0026', '0018']
#zstarget = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
#snap_type = [True,True,True,True,True,True,True,True,True,True,True,True,True]
#snap_files = ['0127', '0102', '0092', '0076', '0064', '0056', '0048', '0040', '0032', '0026', '0018']
#zstarget = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
#snap_type = [True, True, True,True,True, True,True,True,True,True,True,True]
#snap_files = ['0122', '0110', '0106','0100','0098','0096','0094','0088','0084','0080','0072','0068']
#zstarget = [0.05,0.3,0.4,0.6,0.7,0.8,0.9,1.125,1.5,1.75,2.25,2.5]
#snap_type = [True, True, True,True,True, True,True,True,True,True,True,True]

#snap_files = ['0100','0096','0088','0084','0080','0072','0068']
#zstarget = [0.6,0.8,1.125,1.5,1.75,2.25,2.5]
#snap_type = [True,True,True,True,True,True,True,True]


#snap_files = ['0040', '0026', '0018']
#zstarget = [6.0, 8.0, 10.0]
#snap_type = [True,True,True]

snap_files = ['0123', '0088', '0072', '0064', '0060', '0048', '0040'] #, '0026', '0020']
zstarget = [0.0, 1.0, 2.0, 3.0, 3.5, 4.0, 5.0, 6.0] #, 8.0, 10.0]
snap_type = [True,True,True,True,True,True,True,True]


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

#define radial bins of interest. This going from 0 to 50kpc, in bins of 1kpc
rmax = 50
rmin = 0
dr = 1.5
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

def distance_2d_grid_face(x, y, z, coord, spin_vec):
   
    coordin = coord
    #centre particles first
    coordin[:,0] = coordin[:,0] - x
    coordin[:,1] = coordin[:,1] - y
    coordin[:,2] = coordin[:,2] - z

    #calculate magnitude of vectors. 
    magpos = np.sqrt(coordin[:,0]**2 + coordin[:,1]**2 +coordin[:,2]**2)

    #I need to choose a vector that gives me a dot product =0 with the spin (perpendicular to the spin).
    #I will select all star particles that have a dot product close to 0 and choose any of the randomly to define the viewing angle.
    costhetapos=(coordin[:,0] * spin_vec[0] + coordin[:,1] * spin_vec[1] + coordin[:,2] * spin_vec[2]) / magpos[:]
    #select particle that is the closest to perpendicular to the spin:
    plane = np.where(abs(costhetapos) == min(abs(costhetapos))) #make sure cosine is as small as it can be
    VecObs = coordin[plane[0],:] / magpos[plane[0]]
    print(VecObs.shape)
    VecObs = VecObs[0]
    vectorplane = np.zeros(shape = 3)
    vectorplane[0] = (VecObs[1]*spin_vec[2] - VecObs[2]*spin_vec[1])
    vectorplane[1] = (VecObs[2]*spin_vec[0] - VecObs[0]*spin_vec[2])
    vectorplane[2] = (VecObs[0]*spin_vec[1] - VecObs[1]*spin_vec[0])

    #make sure plane vector is normalised
    normvec = np.sqrt(vectorplane[0]**2 + vectorplane[1]**2 + vectorplane[2]**2)
    vectorplane[:] = vectorplane[:] / normvec
    #calculate angle between position vectors and vectorplane, and between velocity and view vector
    costhetaplane = (vectorplane[0] * coordin[:,0] + vectorplane[1] * coordin[:,1] + vectorplane[2] * coordin[:,2]) / magpos[:]

    #calculate positions y and x as detailed above.
    xnew = magpos * costhetaplane
    ynew = magpos * costhetapos
   
    return xnew, ynew

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
    cos_theta = np.sqrt(cross_prod[:,0]**2 + cross_prod[:,1]**2 + cross_prod[:,2]**2)
    sin_theta = np.sin(np.acos(cos_theta))
    #return projected distance
    dcentre3d = distance_3d(x,y,z, coord)
    return dcentre3d * sin_theta, dcentre3d * cos_theta


def v_z_dir(v,spin):
    #funciton assumes spin is normalised, but v isn't.
    v = v[0]
    vz = (v[:,0] * spin[0] +  v[:,1] * spin[1] +  v[:,2] * spin[2])
    return vz
 
##### loop through redshifts ######
for z in range(0,1): # 7,len(snap_files)):
    snap_file =snap_files[z]
    ztarget = zstarget[z]
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
    select = np.where((m30 >= sm_limit) & (sfr30 > 0))
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
       print(v_at_cp.shape, v_at_cp[0,:])
       Jstars_in = Jstars_in[0]
       Jstars_in_norm = np.sqrt(Jstars_in[:,0]**2 + Jstars_in[:,1]**2 + Jstars_in[:,2]**2)
       spin_vec_norm = np.zeros(shape= (ngals,3)) 
       spin_vec_norm[:,0] = Jstars_in[:,0] / Jstars_in_norm #normalise Jstars vector. Needed to find the plane of rotation
       spin_vec_norm[:,1] = Jstars_in[:,1] / Jstars_in_norm #normalise Jstars vector. Needed to find the plane of rotation
       spin_vec_norm[:,2] = Jstars_in[:,2] / Jstars_in_norm #normalise Jstars vector. Needed to find the plane of rotation

   
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
       npart0_cold_profile = np.zeros(shape = (ngals, nr))
       npart4_profile = np.zeros(shape = (ngals, nr))
       disp_HI_profile = np.zeros(shape = (ngals, nr))
       disp_H2_profile = np.zeros(shape = (ngals, nr))
       disp_cool_profile = np.zeros(shape = (ngals, nr))
       disp_HI_profile_h5 = np.zeros(shape = (ngals, nr))
       disp_H2_profile_h5 = np.zeros(shape = (ngals, nr))
       disp_cool_profile_h5 = np.zeros(shape = (ngals, nr))
       disp_HI_profile_h10 = np.zeros(shape = (ngals, nr))
       disp_H2_profile_h10 = np.zeros(shape = (ngals, nr))
       disp_cool_profile_h10 = np.zeros(shape = (ngals, nr))

       r_dist_centre = np.zeros(shape = nr) 
       if (family_method == 'radial_profiles'):
            r_dist_centre = xr
       elif (family_method == 'grid'):
            for i,ri in enumerate(gr):
                for j,rj in enumerate(gr):
                    r_dist_centre[i * len(gr) + j] = np.sqrt(ri**2 + rj**2)
 
       ################################# read particle data #####################################################
       fields = {'PartType0': ('GroupNr_bound', 'Coordinates' , 'Masses', 'StarFormationRates', 'Temperatures', 'Densities', 'Velocities')}
       print("Will read part type 0 properties")
       h5data = common.read_particle_data_colibre(model_dir, snap_file, fields)
       print("Have read particle 0 properties")
       
       #  SpeciesFractions: "elec", "HI", "HII", "Hm", "HeI", "HeII", "HeIII", "H2", "H2p", "H3p" (for snapshots)
       #  ReducedSpeciesFractions: : "HI", "HeI", "HeII", "H2"
       #  ElementMassFractions: "Hydrogen", "Helium", "Carbon", "Nitrogen", "Oxygen", "Neon", "Magnesium", "Silicon", "Iron", "Strontium", "Barium", "Europium"
       #  ReducedElementMassFractions:  "Hydrogen", "Helium" (for snipshots)
       #  SubhaloID is now unique to the whole box, so we only need to match a signel number
       #  (sgnp, coord, m, sfr, temp, speciesfrac, elementmassfracs, elementmassfracsdiff, DustMassFrac, dens) = h5data
       (sgnp, coord, m, sfr, temp, dens, v) = h5data
       del(h5data) #release data

       ################################# read particle data #####################################################
       fields = {'PartType0': ('SpeciesFractions', 'DustMassFractions')}
       print("Will read part type 0 properties")
       h5data = common.read_particle_data_colibre(model_dir, snap_file, fields)
       print("Have read particle 0 properties")
       speciesfrac, DustMassFrac = h5data
       del(h5data) #release data

       if(snap_type[z] == True):
          speciesfrac2 = np.zeros(shape = (len(speciesfrac[:,0]), 2))
          speciesfrac2[:,0] = speciesfrac[:,1]
          speciesfrac2[:,1] = speciesfrac[:,7]
          speciesfrac = speciesfrac2
       DustMassFracTot = DustMassFrac[:,0] + DustMassFrac[:,1] + DustMassFrac[:,2] + DustMassFrac[:,3] + DustMassFrac[:,4] + DustMassFrac[:,5]
       del(speciesfrac2, DustMassFrac)

       fields = {'PartType0': ('ElementMassFractions', 'ElementMassFractionsDiffuse')}
       print("Will read part type 0 properties")
       h5data = common.read_particle_data_colibre(model_dir, snap_file, fields)
       print("Have read particle 0 properties")
       elementmassfracs, elementmassfracsdiff = h5data
       del(h5data) #release data

       def reduce_dim_elements(elementmassfracs):
           els = np.zeros(shape = (len(elementmassfracs[:,0]), 3))
           els[:,0] = elementmassfracs[:,0]
           els[:,1] = elementmassfracs[:,4]
           els[:,2] = elementmassfracs[:,8]
           return els

       elementmassfracs = reduce_dim_elements(elementmassfracs)
       elementmassfracsdiff = reduce_dim_elements(elementmassfracsdiff)

       #keep only particles in the WNM or CNM
       cold = np.where(temp <= 10**(4.5))
       sgnp = sgnp[cold]
       temp = temp[cold]
       m = m[cold]
       coord = coord[cold,:]
       sfr = sfr[cold]
       dens = dens[cold]
       DustMassFracTot = DustMassFracTot[cold]
       elementmassfracs = elementmassfracs[cold,:]
       elementmassfracsdiff = elementmassfracsdiff[cold,:]
       speciesfrac = speciesfrac[cold,:]
       coord = coord[0]
       elementmassfracs = elementmassfracs[0]
       elementmassfracsdiff = elementmassfracsdiff[0]
       speciesfrac = speciesfrac[0]
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

       print("Will read part type 4 properties")
       fields = {'PartType4': ('GroupNr_bound', 'Coordinates' , 'Masses', 'Velocities')}
       h5data = common.read_particle_data_colibre(model_dir, snap_file, fields)
       print("Have read particle 4 properties")
       #SubhaloID is now unique to the whole box, so we only need to match a single number
       (sgnpT4, coordT4, mT4, vT4) = h5data
       del(h5data) #release data
       #fields_in = np.isin(sgnpT4,sgn_in) #find all particles of interest and redefine arrays with only those
       print("Mean x velocity for gas and stars (should be ~0):", np.mean(v[:,0]), np.mean(vT4[:,0]))
       
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

           #select particles type 4 with the same Subhalo ID
           partin4 = np.where(sgnpT4 == sgn_in[g])
           npartingalT4 = len(sgnpT4[partin4])
           if(npartingalT4 > 0):
              coord_in_p4 = coordT4[partin4,:]
              coord_in_p4 = coord_in_p4[0]
              vT4_in = vT4[partin4,:]
              vT4_in = vT4_in[0]
              m_part4 = mT4[partin4]
              #compute mean velocity with all stellar particles within 1kpc.
              dcentre_3d_T4 =  distance_3d(x_in[g], y_in[g], z_in[g], coord_in_p4)
              in1kpc = np.where(dcentre_3d_T4 <= 1e-3)
              vgx = np.mean(vT4_in[in1kpc,0])
              vgy = np.mean(vT4_in[in1kpc,1])
              vgz = np.mean(vT4_in[in1kpc,2])
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
                      dcentre, dheight = distance_2d_faceon(x_in[g], y_in[g], z_in[g], coord_in_p0, spin_vec_norm[g,:])
              elif (family_method == 'grid'):
                  dcentre =  distance_3d(x_in[g], y_in[g], z_in[g], coord_in_p0)
 
              #define vectors with mass and SFR of particles
              m_part0 = m[partin]
              sfr_part0 = sfr[partin]
              v_part0 = v[partin,:]
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
              v_part0 = v_part0[0]
              #move velocities to the reference frame of the galaxy, but first compute the frame with cold gas within 5kpc of the centre
              in5kpc = np.where((dcentre < 5e-3) & (temp_part0 < 1e4))
              vgx = np.mean(v_part0[in5kpc,0])
              vgy = np.mean(v_part0[in5kpc,1])
              vgz = np.mean(v_part0[in5kpc,2])
              print("residual velocities:", vgx-v_at_cp[g,0], vgy-v_at_cp[g,1], vgz-v_at_cp[g,2])

              v_part0[:,0] = v_part0[:,0] - vgx #v_at_cp[g,0] #np.mean(v_part0[:,0]) #
              v_part0[:,1] = v_part0[:,1] - vgy #v_at_cp[g,1] #np.mean(v_part0[:,1]) #
              v_part0[:,2] = v_part0[:,2] - vgz #v_at_cp[g,2] #np.mean(v_part0[:,2]) #
              print("Mean x velocity (should be ~0):", np.mean(v_part0[:,0]), " mean distance in kpc", np.mean(coord_in_p0[:,0]))
 
              if (family_method == 'radial_profiles'):
                  for i,r in enumerate(xr):
                      inr = np.where((dcentre >= (r - dr/2)*1e-3) & (dcentre < (r + dr/2)*1e-3))
                      if(len(dcentre[inr]) > 0):
                         npart0_profile[g,i] = len(dcentre[inr])
                         vr_inr = v_z_dir(v_part0[inr,:], spin_vec_norm[g,:])
                         temp_inr = temp_part0[inr]
                         dens_inr = dens_part0[inr]
                         mh_inr = m_part0[inr] * elementmassfracs_part0[inr,0]
                         mhdiff_inr = m_part0[inr] * elementmassfracsdiff_part0[inr,0]
                         mo_inr = m_part0[inr] * elementmassfracsdiff_part0[inr,1]
                         mfe_inr = m_part0[inr] * elementmassfracsdiff_part0[inr,2]
                         sfr_inr = sfr_part0[inr]
                         mh2_partin = mh_inr * speciesfrac_part0[inr,1] * 2
                         mhi_partin = mh_inr * speciesfrac_part0[inr,0]
                         #reduce dimensionality
                         mh_inr = mh_inr[0]
                         mhdiff_inr = mhdiff_inr[0]
                         mo_inr = mo_inr[0]
                         mfe_inr = mfe_inr[0]
                         mpart_inr = m_part0[inr]
                         mh2_partin = mh2_partin[0]
                         mhi_partin = mhi_partin[0]
                         #calculate profiles
                         sfr_profile[g,i] = np.sum(sfr_inr)
                         mHI_profile[g,i] = np.sum(mh_inr * speciesfrac_part0[inr,0])
                         mH2_profile[g,i] = np.sum(mh_inr * speciesfrac_part0[inr,1] * 2) #factor 2 comes from H2 being two hydrogen atoms
                         mdust_profile[g,i] = np.sum(mpart_inr * dust_part0[inr])
                         disp_H2_profile[g,i] = np.sqrt(np.sum(mh2_partin * vr_inr**2) / mH2_profile[g,i])
                         disp_HI_profile[g,i] = np.sqrt(np.sum(mhi_partin * vr_inr**2) / mHI_profile[g,i])
                         if(method == 'circular_apertures_face_on_map'):
                            dheight_inr = dheight[inr]
                            inh = np.where(abs(dheight_inr )<= 1e-2) #within 10kpc from the disk plane
                            if(len(mh2_partin[inh]) > 0):
                               disp_H2_profile_h10[g,i] = np.sqrt(np.sum(mh2_partin[inh] * vr_inr[inh]**2) / np.sum(mh2_partin[inh]))
                               disp_HI_profile_h10[g,i] = np.sqrt(np.sum(mhi_partin[inh] * vr_inr[inh]**2) / np.sum(mhi_partin[inh]))
                            inh = np.where(abs(dheight_inr )<= 5e-3) #within 10kpc from the disk plane
                            if(len(mh2_partin[inh]) > 0):
                               disp_H2_profile_h5[g,i] = np.sqrt(np.sum(mh2_partin[inh] * vr_inr[inh]**2) / np.sum(mh2_partin[inh]))
                               disp_HI_profile_h5[g,i] = np.sqrt(np.sum(mhi_partin[inh] * vr_inr[inh]**2) / np.sum(mhi_partin[inh]))

                         coldp = np.where((temp_inr < 10**(4.5)) & (dens_inr > 0)) #select particles with temperatures cooler than 10^4.5K and calculate metallicity profiles with those particles only.
                         if(len(mo_inr[coldp]) > 0):
                            mhdiff_inr_cold = mhdiff_inr[coldp]
                            vr_inr_cold = vr_inr[coldp]
                            oh_profile[g,i] = np.sum(mo_inr[coldp]) / np.sum(mhdiff_inr_cold)
                            feh_profile[g,i] = np.sum(mfe_inr[coldp]) / np.sum(mhdiff_inr_cold)
                            coldgas_profile[g,i] = np.sum(mpart_inr[coldp])
                            npart0_cold_profile[g,i] = len(mo_inr[coldp]) 
                            disp_cool_profile[g,i] = np.sqrt(np.sum(mhdiff_inr_cold * vr_inr_cold**2) / np.sum(mhdiff_inr_cold))

                            if(method == 'circular_apertures_face_on_map'):
                               dheight_cold = dheight_inr[coldp]
                               inh = np.where(abs(dheight_cold) <= 1e-2) #within 10kpc from the disk plane
                               if(len(dheight_cold[inh]) > 0):
                                    disp_cool_profile_h10[g,i] = np.sqrt(np.sum(mhdiff_inr_cold[inh] * vr_inr_cold[inh]**2) / np.sum(mhdiff_inr_cold[inh]))
                               inh = np.where(abs(dheight_cold) <= 5e-3) #within 5kpc from the disk plane
                               if(len(dheight_cold[inh]) > 0):
                                    disp_cool_profile_h5[g,i] = np.sqrt(np.sum(mhdiff_inr_cold[inh] * vr_inr_cold[inh]**2) / np.sum(mhdiff_inr_cold[inh]))

                         del(temp_inr, dens_inr, mh_inr, mhdiff_inr, mo_inr, mfe_inr, sfr_inr, coldp) #release data

              elif (family_method == 'grid'):
                  dcentre_i = np.zeros(shape = len(coord_in_p0))
                  dcentre_j = np.zeros(shape = len(coord_in_p0))
                  if(method == 'grid_random_map'):
                      dcentre_i, dcentre_j = distance_2d_grid_random(x_in[g], y_in[g], coord_in_p0)
                  if(method == 'grid_face_on_map'):
                      dcentre_i, dcentre_j = distance_2d_grid_face(x_in[g], y_in[g], z_in[g], coord_in_p0, spin_vec_norm[g,:])

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
                          del(temp_inr,dens_inr,mh_inr,mhdiff_inr,mo_inr,mfe_inr,speciesfrac_inr,dust_part_inr,sfr_inr,dcentre_j_inr)
                  del(dcentre_i, dcentre_j)
              del(m_part0, sfr_part0, temp_part0, dust_part0, elementmassfracs_part0, elementmassfracsdiff_part0, dens_part0, speciesfrac_part0) #release data
      
           if(npartingalT4 > 0):
              if (family_method == 'radial_profiles'):
                   #calculate distance between particle and centre of potential
                   if (method == 'spherical_apertures'):
                       dcentre =  distance_3d(x_in[g], y_in[g], z_in[g], coord_in_p4)
                   elif (method == 'circular_apertures_random_map'):
                       dcentre = distance_2d_random(x_in[g], y_in[g], coord_in_p4)
                   elif (method == 'circular_apertures_face_on_map'):
                       dcentre, dheight = distance_2d_faceon(x_in[g], y_in[g], z_in[g], coord_in_p4, spin_vec_norm[g,:])
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
                 if(method == 'grid_face_on_map'):
                     dcentre_i, dcentre_j = distance_2d_grid_face(x_in[g], y_in[g], z_in[g], coord_in_p4, spin_vec_norm[g,:])

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
       np.savetxt('Runs/' + model_name + '/ProcessedData/' +  'Disp_HI_profiles_ap50ckpc_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + ".txt", disp_HI_profile)
       np.savetxt('Runs/' + model_name + '/ProcessedData/' +  'Disp_H2_profiles_ap50ckpc_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + ".txt", disp_H2_profile)
       np.savetxt('Runs/' + model_name + '/ProcessedData/' +  'Disp_Cool_profiles_ap50ckpc_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + ".txt", disp_cool_profile)
       if(method == 'circular_apertures_face_on_map'):
           np.savetxt('Runs/' + model_name + '/ProcessedData/' +  'Disp_HI_h10_profiles_ap50ckpc_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + ".txt", disp_HI_profile_h10)
           np.savetxt('Runs/' + model_name + '/ProcessedData/' +  'Disp_H2_h10_profiles_ap50ckpc_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + ".txt", disp_H2_profile_h10)
           np.savetxt('Runs/' + model_name + '/ProcessedData/' +  'Disp_Cool_h10_profiles_ap50ckpc_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + ".txt", disp_cool_profile_h10)
           np.savetxt('Runs/' + model_name + '/ProcessedData/' +  'Disp_HI_h5_profiles_ap50ckpc_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + ".txt", disp_HI_profile_h5)
           np.savetxt('Runs/' + model_name + '/ProcessedData/' +  'Disp_H2_h5_profiles_ap50ckpc_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + ".txt", disp_H2_profile_h5)
           np.savetxt('Runs/' + model_name + '/ProcessedData/' +  'Disp_Cool_h5_profiles_ap50ckpc_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + ".txt", disp_cool_profile_h5)
   
       #save radii info
       np.savetxt('Runs/' + model_name + '/ProcessedData/' +  'radii_info_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + ".txt", r_dist_centre)
   
