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

input_filename = f"../raw/Bigiel2010.txt"

processed = ObservationalData()

comment = "Outer regions of galaxies"
citation = "Bigiel et al. (2010)"
bibcode = "2010AJ....140.1194B"
name = "Spatially-resolved $\\Sigma_{\\rm HI+H_2}$ vs $\\Sigma_{\\rm SFR}$"
plot_as = "points"

# Reading the bigiel 2010 data

with open(input_filename) as f:
    lines = f.readlines()

size_array = len(lines) - 46

sigma_HI = -5.2 * np.ones(size_array)
sigma_HI_err = -5.2 * np.ones(size_array)
sigma_SFR = 10 ** -0.2 * np.ones(size_array)
sigma_SFR_err = 10 ** -0.2 * np.ones(size_array)

for i in range(49, len(lines)):
    k = i - 49
    word1 = lines[i][20:25]
    word2 = lines[i][26:30]
    word3 = lines[i][31:37]
    word4 = lines[i][38:42]
    if word1 != "     ":
        sigma_HI[k] = float(word1)
    if word2 != "    ":
        sigma_HI_err[k] = float(word2)
    if word3 != "      ":
        sigma_SFR[k] = float(word3)
    if word4 != "    ":
        sigma_SFR_err[k] = float(word4)

sigma_gas = 10 ** sigma_HI / 1.36  # a factor of 1.36 to account for heavy elements

array_of_interest = np.arange(-1, 3, 0.25)
minimum_surface_density = 0.4  # the paper quotes 0.5 Msun/pc^2, but this seems to be much more than that is possible given the Sigma_SFR cut, so the value it reaches this is around 0.4
array_of_interest = array_of_interest[array_of_interest >= minimum_surface_density]
if array_of_interest[0] > minimum_surface_density:
    array_of_interest = np.append([minimum_surface_density], array_of_interest)

Obs_Hneutral = sigma_gas

Obs_SFR = sigma_SFR * 1e-5

binned_data = bin_data_general(
    np.log10(Obs_Hneutral), np.log10(Obs_SFR), array_of_interest
)

SigmaHneutral = unyt.unyt_array(10 ** binned_data[0], units="Msun/pc**2")

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

SigmaHneutral_err = unyt.unyt_array(
    [
        10 ** (binned_data[0]) - 10 ** (binned_data[0] - array_x_bin_std_down),
        10 ** (binned_data[0] + array_x_bin_std_up) - 10 ** (binned_data[0]),
    ],
    units="Msun/pc**2",
)

processed.associate_x(
    SigmaHneutral,
    scatter=SigmaHneutral_err,
    comoving=False,
    description="H2 + HI Surface density",
)

processed.associate_y(
    SigmaSFR,
    scatter=SigmaSFR_err,
    comoving=False,
    description="Star Formation Rate Surface Density",
)

processed.associate_citation(citation, bibcode)
processed.associate_name(name)
processed.associate_comment(comment)
processed.associate_redshift(0.0, 0.0, 0.0)
processed.associate_plot_as(plot_as)
processed.associate_cosmology(cosmology)

output_path = f"../Bigiel2010.hdf5"

if os.path.exists(output_path):
    os.remove(output_path)

processed.write(filename=output_path)
