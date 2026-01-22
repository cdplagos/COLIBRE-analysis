

import numpy as np
import os 
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
#model_name = 'L0100N1504/Thermal/'
#model_name = 'L0050N0752/Thermal/'
#model_name = 'L0025N0376/Thermal/'
#model_name = 'L0025N0188/Thermal/'
#model_name = 'L0025N0752/Thermal/'
model_name = 'L200_m6/Thermal/'

model_dir = '/cosma8/data/dp004/colibre/Runs/' + model_name
out_dir = '/cosma8/data/dp004/ngdg66/Runs/' + model_name
dir_output_data = '/ProcessedData/'

files_to_combine = ['Mstar_SFH_ap50ckpc_']


ztarget = 0.0
dt = 0.01
subv = range(0,640)

ids = np.loadtxt(out_dir + dir_output_data +  'GalaxyProperties_sfrGE0_z' + str(ztarget) + '.txt', unpack = True, usecols = [0])
rbin = np.loadtxt(out_dir + dir_output_data +  'look_back_time_info_' + method + "_dt" + str(dt) + "_z" + str(ztarget) + ".txt")
ngals = len(ids)
nrbins = len(rbin)

#loop through files to combine
for j, f in enumerate(files_to_combine):
    init = 0
    #loop through subvolumes
    for i in subv:
       #check if the file isn't empty
       if(os.path.getsize(out_dir + dir_output_data +  f + method + "_dt"+ str(dt) + "_z" + str(ztarget) + "subvolume_" + str(i) +  ".txt") != 0):
          #read galaxy IDs and profiles that will need to be concatenate.
          ids_in_subv = np.loadtxt(out_dir + dir_output_data +  'Galaxies_sfrGE0_in_subv_z' + str(ztarget) + "subvolume_" + str(i) + ".txt")
          data_file = np.loadtxt(out_dir + dir_output_data +  f + method + "_dt"+ str(dt) + "_z" + str(ztarget) + "subvolume_" + str(i) +  ".txt")
          #print("reading subv", i)
          #in the case there is a single galaxy in this suvolume, make sure the data_file array is a matrix for concatenation later
          if(ids_in_subv.size == 1):
              data_file = np.array([data_file])

          #check if this is the first subvolume that is read and isn't empty
          if(init == 0):
             #initialize arrays for concatenation
             data = data_file
             ids_all = ids_in_subv
             init = 1
          else:
             #concatenate data
             data = np.append(data, data_file, axis=0)
             ids_all = np.append(ids_all, ids_in_subv)
            
    #the difference between the two numbers being printed are the galaxies that are split between subvolumes.
    print("Galaxies in read subvolumes", len(np.unique(ids_all)), len(ids_all))
    data_to_save = np.zeros(shape = (ngals, nrbins))

    #now loop through all the galaxies that were selected in this snapshot
    for g, idx in enumerate(ids):
        match = np.where(ids_all == idx)
        #if there is more than one match, we need to correct the profiles (this happens when particles of a single galaxy are split between subvolume files)
        if(len(ids_all[match]) > 1):
           rowsall = data[match,:]
           rowsall = rowsall[0]
           mgastot = np.zeros(shape = nrbins)
           #loop through matches to correct the quantities in this galaxy
           for m in range(0,len(ids_all[match])):
              #simply sum the masses
              data_to_save[g,:] = data_to_save[g,:] + rowsall[m,:]
        #if there us a single match, simply save the profile of this galaxy
        if(len(ids_all[match]) == 1):
           data_to_save[g,:] = data[match[0],:]

    #save unified file
    print("Will save file", out_dir + dir_output_data +  f + method + "_dt"+ str(dt) + "_z" + str(ztarget) + ".txt")    
    np.savetxt(out_dir + dir_output_data +  f + method + "_dt"+ str(dt) + "_z" + str(ztarget) + ".txt", data_to_save)


