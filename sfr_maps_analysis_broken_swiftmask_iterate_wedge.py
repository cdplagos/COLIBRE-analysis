import os
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import common
#import utilities_statistics as us
from swiftsimio import load as swiftsimio_loader
from swiftsimio import mask as swiftsimio_mask

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
boxsize = 200.0

model_dir = '/cosma8/data/dp004/colibre/Runs/' + model_name
out_dir = '/cosma8/data/dp004/ngdg66/Runs/' + model_name 

sm_limit = 1e9

#definitions below correspond to z=0
snap_files = ['0127', '0119', '0114', '0102', '0092', '0076', '0064', '0056', '0048', '0040', '0032', '0026', '0018']
zstarget = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
snap_type = [True,True,True,True,True,True,True,True,True,True,True,True,True]
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

#snap_files = ['0123', '0088', '0072', '0064', '0060', '0048', '0040'] #, '0026', '0020']
#zstarget = [0.0, 1.0, 2.0, 3.0, 3.5, 4.0, 5.0, 6.0] #, 8.0, 10.0]
#snap_type = [True,True,True,True,True,True,True,True]

#snap_files = ['0127','0102', '0092', '0076', '0064', '0056', '0048', '0040', '0026', '0018']
#zstarget = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
#snap_type = [True, True,True,True,True,True,True,True,True,True]


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


#volume limits for SWISIMIO mask. Needs to be in units of the volume
#ivol = np.array([0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]) * boxsize
ivol = np.array([0, 0.25, 0.5, 0.75, 1.0]) * boxsize
len_wedge = (len(ivol) - 1)**3
xlimits = np.zeros(shape = (len_wedge,2))
ylimits = np.zeros(shape = (len_wedge,2))
zlimits = np.zeros(shape = (len_wedge,2))

#loop through wedges 
w = 0
for xi in range(0, len(ivol)-1):
    for yi in range(0, len(ivol)-1):
        for zi in range(0, len(ivol)-1):
            xlimits[w,:] = [ivol[xi], ivol[xi+1]]
            ylimits[w,:] = [ivol[yi], ivol[yi+1]]
            zlimits[w,:] = [ivol[zi], ivol[zi+1]]
            
            if(w == 0):
               first_wedge = True
            else:
               first_wedge = False
            wedge_name = '_wedge_%s' % str(w)
            print("analysing wedge", xlimits[w,:], ylimits[w,:], zlimits[w,:], wedge_name, first_wedge)

            ##### loop through redshifts ######
            for z in range(0, 1): #8,len(snap_files)):
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
                   Jstars_in = Jstars_in[0]
                   Jstars_in_norm = np.sqrt(Jstars_in[:,0]**2 + Jstars_in[:,1]**2 + Jstars_in[:,2]**2)
                   spin_vec_norm = np.zeros(shape= (ngals,3)) 
                   spin_vec_norm[:,0] = Jstars_in[:,0] / Jstars_in_norm #normalise Jstars vector. Needed to find the plane of rotation
                   spin_vec_norm[:,1] = Jstars_in[:,1] / Jstars_in_norm #normalise Jstars vector. Needed to find the plane of rotation
                   spin_vec_norm[:,2] = Jstars_in[:,2] / Jstars_in_norm #normalise Jstars vector. Needed to find the plane of rotation
            
                   if(first_wedge):
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
                      np.savetxt(out_dir + '/ProcessedData/' + 'GalaxyProperties_z' + str(ztarget) + '.txt', gal_props)
                      print("Have saved galaxy properties") 
                      del(gal_props) #releasing memory
             
                   #initialise profile arrays
                   sfr_profile = np.zeros(shape = (ngals, nr))
                   mHI_profile = np.zeros(shape = (ngals, nr))
                   mH2_profile = np.zeros(shape = (ngals, nr))
                   mstar_profile = np.zeros(shape = (ngals, nr))
                   oh_profile = np.zeros(shape = (ngals, nr))
                   #feh_profile = np.zeros(shape = (ngals, nr))
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
                   galaxies_in_wedge = np.zeros(shape = ngals)
            
                   r_dist_centre = np.zeros(shape = nr) 
                   if (family_method == 'radial_profiles'):
                        r_dist_centre = xr
                   elif (family_method == 'grid'):
                        for i,ri in enumerate(gr):
                            for j,rj in enumerate(gr):
                                r_dist_centre[i * len(gr) + j] = np.sqrt(ri**2 + rj**2)
            
                   path = os.path.join(model_dir, 'SOAP-HBT', 'colibre_with_SOAP_membership_' + snap_file + '.hdf5')
                   pdata_snap = swiftsimio_loader(path)
                   mask = swiftsimio_mask(path)
                   mask.constrain_spatial([xlimits[w,:], ylimits[w,:], zlimits[w,:]])
                   pdata_snap_masked = swiftsimio_loader(path, mask=mask)
             
                   print("Will read part type 0 properties")
                   pdata_masked_object = pdata_snap_masked.gas 
                   temp = pdata_masked_object.temperatures
                   temp = temp.value
                   print("read temperature") 
                   dens = pdata_masked_object.densities
                   dens = dens.value
                   print("read densities")
                   v = pdata_masked_object.velocities
                   v = v.value
                   print("read velocities")
                   coord = pdata_masked_object.coordinates
                   coord = coord.value
                   print("read coordinates", min(coord[:,0]), max(coord[:,0]))
                   m = pdata_masked_object.masses
                   m = m.value
                   print("read masses")
                   sfr = pdata_masked_object.star_formation_rates
                   sfr= sfr.value
                   print("read star formation rates")
                   sgnp = pdata_masked_object.group_nr_bound
                   sgnp = sgnp.value
                   print("read group number bound")
                   speciesfracHI = pdata_masked_object.species_fractions
                   speciesfrac = np.zeros(shape = (len(temp),2))
                   speciesfrac[:,0] = speciesfracHI.HI.value
                   speciesfrac[:,1] = speciesfracHI.H2.value
                   del speciesfracHI.HI, speciesfracHI.H2
                   print("read species fractions")
                   DustMassFrac = pdata_masked_object.dust_mass_fractions
                   DustMassFracTot = DustMassFrac.GraphiteLarge.value + DustMassFrac.MgSilicatesLarge.value + DustMassFrac.FeSilicatesLarge.value + DustMassFrac.GraphiteSmall.value + DustMassFrac.MgSilicatesSmall.value+ DustMassFrac.FeSilicatesSmall.value
                   del DustMassFrac.GraphiteLarge, DustMassFrac.MgSilicatesLarge, DustMassFrac.FeSilicatesLarge, DustMassFrac.GraphiteSmall, DustMassFrac.MgSilicatesSmall, DustMassFrac.FeSilicatesSmall
                   print("read dust mass fractions")
                   elementmassfracs = pdata_masked_object.element_mass_fractions.hydrogen
                   elementmassfracs = elementmassfracs.value
                   print("read element mass fractions")
            
                   elementmassfracsdiff_h = pdata_masked_object.element_mass_fractions_diffuse.hydrogen
                   elementmassfracsdiff_o = pdata_masked_object.element_mass_fractions_diffuse.oxygen
                   #elementmassfracsdiff_f = pdata_masked_object.element_mass_fractions_diffuse.iron
            
                   elementmassfracsdiff = np.zeros(shape = (len(temp),2))
                   elementmassfracsdiff[:,0] = elementmassfracsdiff_h.value
                   elementmassfracsdiff[:,1] = elementmassfracsdiff_o.value
                   #elementmassfracsdiff[:,2] = elementmassfracsdiff_f.value
                   del elementmassfracsdiff_h, elementmassfracsdiff_o #, elementmassfracsdiff_f
                   print("read element mass fractions diffuse")
                   print("Have read particle 0 properties")
                   
                   #keep only particles in the WNM or CNM
                   cold = np.where(temp <= 10**(4.5))
                   sgnp = sgnp[cold]
                   temp = temp[cold]
                   m = m[cold]
                   coord = coord[cold,:]
                   print("min and max coord gas x", min(coord[:,0]), max(coord[:,0]))
                   sfr = sfr[cold]
                   dens = dens[cold]
                   DustMassFracTot = DustMassFracTot[cold]
                   elementmassfracs = elementmassfracs[cold]
                   elementmassfracsdiff = elementmassfracsdiff[cold,:]
                   speciesfrac = speciesfrac[cold,:]
                   coord = coord[0]
                   elementmassfracsdiff = elementmassfracsdiff[0]
                   speciesfrac = speciesfrac[0]
            
                   print("Will read part type 4 properties")
                   pdata_masked_object = pdata_snap_masked.stars 
                   sgnpT4 = pdata_masked_object.group_nr_bound
                   sgnpT4 = sgnpT4.value
                   print("read group number bound")
                   coordT4 = pdata_masked_object.coordinates
                   coordT4 = coordT4.value
                   print("read coordinates")
                   mT4 = pdata_masked_object.masses
                   mT4 = mT4.value
                   print("read masses")
                   vT4 = pdata_masked_object.velocities
                   vT4 = vT4.value
                   print("read velocities")
                   print("Have read particle 4 properties")
            
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
                          galaxies_in_wedge[g] = sgn_in[g]
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
                          elementmassfracs_part0 = elementmassfracs[partin]
                          elementmassfracsdiff_part0 = elementmassfracsdiff[partin,:]
                          dens_part0 = dens[partin]
                          speciesfrac_part0 = speciesfrac[partin,:]
                          #reduce dimensionality
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
                                     mh_inr = m_part0[inr] * elementmassfracs_part0[inr]
                                     mhdiff_inr = m_part0[inr] * elementmassfracsdiff_part0[inr,0]
                                     mo_inr = m_part0[inr] * elementmassfracsdiff_part0[inr,1]
                                     #mfe_inr = m_part0[inr] * elementmassfracsdiff_part0[inr,2]
                                     sfr_inr = sfr_part0[inr]
                                     mh2_partin = mh_inr * speciesfrac_part0[inr,1] * 2
                                     mhi_partin = mh_inr * speciesfrac_part0[inr,0]
                                     #reduce dimensionality
                                     mh_inr = mh_inr[0]
                                     mhdiff_inr = mhdiff_inr[0]
                                     mo_inr = mo_inr[0]
                                     #mfe_inr = mfe_inr[0]
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
                                        #feh_profile[g,i] = np.sum(mfe_inr[coldp]) / np.sum(mhdiff_inr_cold)
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
            
                                     del(temp_inr, dens_inr, mh_inr, mhdiff_inr, mo_inr, sfr_inr, coldp) #mfe_inr, sfr_inr, coldp) #release data
            
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
                                      mh_inr = m_part0[inr] * elementmassfracs_part0[inr]
                                      mhdiff_inr = m_part0[inr] * elementmassfracsdiff_part0[inr,0]
                                      mo_inr = m_part0[inr] * elementmassfracsdiff_part0[inr,1]
                                      #mfe_inr = m_part0[inr] * elementmassfracsdiff_part0[inr,2]
                                      speciesfrac_inr = speciesfrac_part0[inr,:]
                                      dust_part_inr = dust_part0[inr]
                                      #reduce dimensionality
                                      mh_inr = mh_inr[0]
                                      mhdiff_inr = mhdiff_inr[0]
                                      mo_inr = mo_inr[0]
                                      #mfe_inr = mfe_inr[0]
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
                                              #mfe_inj = mfe_inr[inrj] 
                                              mpart_inj = mpart_inr[inrj]
                                              coldp = np.where((temp_inj < 10**(4.5)) & (dens_inj > 0)) #select particles with temperatures cooler than 10^4.5K and calculate metallicity profiles with those particles only.
                                              if(len(mo_inj[coldp]) > 0):
                                                 oh_profile[g,i * len(gr) + j] = np.sum(mo_inj[coldp]) / np.sum(mh_inj[coldp])
                                                 #feh_profile[g,i * len(gr) + j] = np.sum(mfe_inj[coldp]) / np.sum(mh_inj[coldp])
                                                 coldgas_profile[g,i * len(gr) + j] = np.sum(mpart_inj[coldp])
                                              del(temp_inj,dens_inj,mh_inj,mo_inj,mpart_inj,coldp) #mfe_inj,mpart_inj,coldp)
                                      del(temp_inr,dens_inr,mh_inr,mhdiff_inr,mo_inr,speciesfrac_inr,dust_part_inr,sfr_inr,dcentre_j_inr) #mfe_inr
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
            
            
                   #select galaxies that were processed:
                   inw = np.where(galaxies_in_wedge != 0)
                   print("Number of galaxies processed in this wedge", len(galaxies_in_wedge[inw]), " out of", ngals) 
                   #save galaxy profiles
                   sfrin = sfr_profile[inw,:]
                   mHIin = mHI_profile[inw,:]
                   mH2in = mH2_profile[inw,:]
                   ohin  = oh_profile[inw,:]
                   #fehin = feh_profile[inw,:]
                   msin  = mstar_profile[inw,:] 
                   mdin  = mdust_profile[inw,:]
                   np0in = npart0_profile[inw,:]
                   np4in = npart4_profile[inw,:]
                   coldgin = coldgas_profile[inw,:]
                   sHIin = disp_HI_profile[inw,:]
                   sH2in = disp_H2_profile[inw,:]
                   scoolin = disp_cool_profile[inw,:]
                   np.savetxt(out_dir + '/ProcessedData/' + 'Galaxies_in_wedge_z' + str(ztarget) + wedge_name + ".txt", galaxies_in_wedge[inw])
                   np.savetxt(out_dir + '/ProcessedData/' + 'SFR_profiles_ap50ckpc_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + wedge_name + ".txt", sfrin[0])
                   np.savetxt(out_dir + '/ProcessedData/' +  'MHI_profiles_ap50ckpc_'+ method + "_dr"+ str(dr) + "_z"  + str(ztarget) + wedge_name + ".txt", mHIin[0])
                   np.savetxt(out_dir + '/ProcessedData/' +  'MH2_profiles_ap50ckpc_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + wedge_name + ".txt", mH2in[0])
                   np.savetxt(out_dir + '/ProcessedData/' +  'OH_gas_profiles_ap50ckpc_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + wedge_name + ".txt", ohin[0])
                   #np.savetxt(out_dir + '/ProcessedData/' +  'FeH_gas_profiles_ap50ckpc_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + wedge_name + ".txt", fehin[0])
                   np.savetxt(out_dir + '/ProcessedData/' +  'Mstar_profiles_ap50ckpc_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + wedge_name + ".txt", msin[0])
                   np.savetxt(out_dir + '/ProcessedData/' +  'Mdust_profiles_ap50ckpc_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + wedge_name + ".txt", mdin[0])
                   np.savetxt(out_dir + '/ProcessedData/' +  'NumberPart0_profiles_ap50ckpc_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + wedge_name + ".txt", np0in[0])
                   np.savetxt(out_dir + '/ProcessedData/' +  'NumberPart4_profiles_ap50ckpc_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + wedge_name + ".txt", np4in[0])
                   np.savetxt(out_dir + '/ProcessedData/' +  'Mcoldgas_profiles_ap50ckpc_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + wedge_name + ".txt", coldgin[0])
                   np.savetxt(out_dir + '/ProcessedData/' +  'Disp_HI_profiles_ap50ckpc_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + wedge_name + ".txt", sHIin[0])
                   np.savetxt(out_dir + '/ProcessedData/' +  'Disp_H2_profiles_ap50ckpc_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + wedge_name + ".txt", sH2in[0])
                   np.savetxt(out_dir + '/ProcessedData/' +  'Disp_Cool_profiles_ap50ckpc_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + wedge_name + ".txt", scoolin[0])
                   if(method == 'circular_apertures_face_on_map'):
                       sHIh10 = disp_HI_profile_h10[inw,:]
                       sH2h10 = disp_H2_profile_h10[inw,:]
                       scoolh10 = disp_cool_profile_h10[inw,:]
                       sHIh5 = disp_HI_profile_h5[inw,:]
                       sH2h5 = disp_H2_profile_h5[inw,:]
                       scoolh5 = disp_cool_profile_h5[inw,:]
            
                       np.savetxt(out_dir + '/ProcessedData/' +  'Disp_HI_h10_profiles_ap50ckpc_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + wedge_name + ".txt", sHIh10[0])
                       np.savetxt(out_dir + '/ProcessedData/' +  'Disp_H2_h10_profiles_ap50ckpc_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + wedge_name + ".txt", sH2h10[0])
                       np.savetxt(out_dir + '/ProcessedData/' +  'Disp_Cool_h10_profiles_ap50ckpc_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + wedge_name + ".txt", scoolh10[0])
                       np.savetxt(out_dir + '/ProcessedData/' +  'Disp_HI_h5_profiles_ap50ckpc_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + wedge_name + ".txt", sHIh5[0])
                       np.savetxt(out_dir + '/ProcessedData/' +  'Disp_H2_h5_profiles_ap50ckpc_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + wedge_name + ".txt", sH2h5[0])
                       np.savetxt(out_dir + '/ProcessedData/' +  'Disp_Cool_h5_profiles_ap50ckpc_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + wedge_name + ".txt", scoolh5[0])
               
                   #save radii info
                   if(first_wedge):
                      np.savetxt(out_dir + '/ProcessedData/' +  'radii_info_' + method + "_dr"+ str(dr) + "_z" + str(ztarget) + ".txt", r_dist_centre)

            #move to the next wedge
            w = w + 1

