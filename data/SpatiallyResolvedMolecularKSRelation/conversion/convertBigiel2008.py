from velociraptor.observations.objects import ObservationalData

import unyt
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import scipy.stats as stats


def bin_data_general(array_x, array_y, array_x_bin):
    array_x_bin_centres = 0.5 * (array_x_bin[1:] + array_x_bin[:-1])
    y_array_bin, _, _ = stats.binned_statistic(
        array_x, array_y, statistic="median", bins=array_x_bin
    )
    y_array_bin_std_up, _, _ = stats.binned_statistic(
        array_x, array_y, statistic=lambda x: np.percentile(x, 84.0), bins=array_x_bin
    )
    y_array_bin_std_down, _, _ = stats.binned_statistic(
        array_x, array_y, statistic=lambda x: np.percentile(x, 16.0), bins=array_x_bin
    )

    y_array_bin_std_up = y_array_bin_std_up - y_array_bin
    y_array_bin_std_down = y_array_bin - y_array_bin_std_down

    return array_x_bin_centres, y_array_bin, y_array_bin_std_down, y_array_bin_std_up


# Exec the master cosmology file passed as first argument
with open(sys.argv[1], "r") as handle:
    exec(handle.read())

input_filename = f"../raw/Bigiel2008.txt"

processed = ObservationalData()

comment = "Inner regions of galaxies"
citation = "Bigiel et al. (2008)"
bibcode = "2008AJ....136.2846B"
name = "Spatially-resolved $\\Sigma_{\\rm H_2}$ vs $\\Sigma_{\\rm SFR}$"
plot_as = "points"

# Reading the bigiel 2008 data

with open(input_filename) as f:
    lines = f.readlines()

size_array = len(lines) - 49

sigma_H2 = -5 * np.ones(size_array)
sigma_H2_err = -5 * np.ones(size_array)
sigma_SFR = -5 * np.ones(size_array)
sigma_SFR_err = -5 * np.ones(size_array)

for i in range(49, len(lines)):
    k = i - 49
    word1 = lines[i][25:29]
    word2 = lines[i][30:34]
    word3 = lines[i][35:39]
    word4 = lines[i][40:44]
    word5 = lines[i][45:50]
    word6 = lines[i][52:56]
    if word3 != "    ":
        sigma_H2[k] = float(word3)
    if word4 != "    ":
        sigma_H2_err[k] = float(word4)
    if word5 != "     ":
        sigma_SFR[k] = float(word5)
    if word6 != "    ":
        sigma_SFR_err[k] = float(word6)

array_of_interest = np.arange(-1, 3, 0.25)
minimum_surface_density = 0.4  # This is slightly lower than their quoted senstivity but they seem to overestimate the error, this is around the limit of noise for Sigma_SFR
array_of_interest = array_of_interest[array_of_interest >= minimum_surface_density]
if array_of_interest[0] > minimum_surface_density:
    array_of_interest = np.append([minimum_surface_density], array_of_interest)

Obs_H2 = 10 ** sigma_H2 / 1.36  # a factor of 1.36 to account for heavy elements

Obs_SFR = 10 ** sigma_SFR

binned_data = bin_data_general(np.log10(Obs_H2), np.log10(Obs_SFR), array_of_interest)

SigmaH2 = unyt.unyt_array(10 ** binned_data[0], units="Msun/pc**2")

SigmaSFR = unyt.unyt_array(10 ** binned_data[1], units="Msun/yr/kpc**2")

SigmaSFR_err = unyt.unyt_array(
    [
        np.abs(10 ** (binned_data[1]) - 10 ** (binned_data[1] - binned_data[2])),
        np.abs(10 ** (binned_data[1] + binned_data[3]) - 10 ** (binned_data[1])),
    ],
    units="Msun/yr/kpc**2",
)

array_x_bin_std_up = array_of_interest[1:] - binned_data[0]
array_x_bin_std_down = binned_data[0] - array_of_interest[:-1]

SigmaH2_err = unyt.unyt_array(
    [
        10 ** (binned_data[0]) - 10 ** (binned_data[0] - array_x_bin_std_down),
        10 ** (binned_data[0] + array_x_bin_std_up) - 10 ** (binned_data[0]),
    ],
    units="Msun/pc**2",
)

processed.associate_x(
    SigmaH2, scatter=SigmaH2_err, comoving=False, description="$\\Sigma_{\\rm H_2}$"
)

processed.associate_y(
    SigmaSFR, scatter=SigmaSFR_err, comoving=False, description="$\\Sigma_{\\rm SFR}$"
)

processed.associate_citation(citation, bibcode)
processed.associate_name(name)
processed.associate_comment(comment)
processed.associate_redshift(0.0, 0.0, 0.0)
processed.associate_plot_as(plot_as)
processed.associate_cosmology(cosmology)

output_path = f"../Bigiel2008.hdf5"

if os.path.exists(output_path):
    os.remove(output_path)

processed.write(filename=output_path)
