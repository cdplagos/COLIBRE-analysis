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

input_filename1 = f"../raw/data_querejeta_bar.txt"
input_filename2 = f"../raw/data_querejeta_center.txt"
input_filename3 = f"../raw/data_querejeta_disk.txt"
input_filename4 = f"../raw/data_querejeta_interarm.txt"
input_filename5 = f"../raw/data_querejeta_spiral_arms.txt"

processed = ObservationalData()

comment = "Based on the PHANGS sample"
citation = "Querejeta et al. (2021)"
bibcode = "2021A&A...656A.133Q"
name = "Spatially-resolved $\\Sigma_{\\rm H_2}$ vs $\\Sigma_{\\rm SFR}$"
plot_as = "points"

# Reading the Ellison 2020 data

array_of_interest = np.arange(-1, 3, 0.25)
minimum_surface_density = (
    0.0  # This seems to be the limit in the paper, however, nothing is said about it
)
array_of_interest = array_of_interest[array_of_interest >= minimum_surface_density]
if array_of_interest[0] > minimum_surface_density:
    array_of_interest = np.append([minimum_surface_density], array_of_interest)

sigma_SFR_bar, sigma_H2_bar = np.genfromtxt(
    input_filename1, unpack=True, usecols=(7, 9), comments="#"
)

sigma_SFR_center, sigma_H2_center = np.genfromtxt(
    input_filename2, unpack=True, usecols=(7, 9), comments="#"
)

sigma_SFR_disk, sigma_H2_disk = np.genfromtxt(
    input_filename3, unpack=True, usecols=(7, 9), comments="#"
)

sigma_SFR_interarm, sigma_H2_interarm = np.genfromtxt(
    input_filename4, unpack=True, usecols=(7, 9), comments="#"
)

sigma_SFR_spiral_arms, sigma_H2_spiral_arms = np.genfromtxt(
    input_filename5, unpack=True, usecols=(7, 9), comments="#"
)

sigma_SFR_total = np.append(sigma_SFR_bar, sigma_SFR_center)
sigma_SFR_total = np.append(sigma_SFR_total, sigma_SFR_disk)
sigma_SFR_total = np.append(sigma_SFR_total, sigma_SFR_interarm)
sigma_SFR_total = np.append(sigma_SFR_total, sigma_SFR_spiral_arms)

sigma_H2_total = np.append(sigma_H2_bar, sigma_H2_center)
sigma_H2_total = np.append(sigma_H2_total, sigma_H2_disk)
sigma_H2_total = np.append(sigma_H2_total, sigma_H2_interarm)
sigma_H2_total = np.append(sigma_H2_total, sigma_H2_spiral_arms)

sigma_SFR_total[~np.isfinite(sigma_SFR_total)] = 1e-20
sigma_SFR_total[sigma_SFR_total <= 0] = 1e-20
sigma_H2_total[~np.isfinite(sigma_H2_total)] = 1e-20
sigma_H2_total[sigma_H2_total <= 0] = 1e-20

Sigma_H2 = sigma_H2_total / 1.36  # a factor of 1.36 to account for heavy elements
Sigma_SFR = sigma_SFR_total

binned_data = bin_data_general(
    np.log10(Sigma_H2), np.log10(Sigma_SFR), array_of_interest
)

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
    SigmaH2, scatter=SigmaH2_err, comoving=False, description="H2 Surface density"
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

output_path = f"../Querejeta2021.hdf5"

if os.path.exists(output_path):
    os.remove(output_path)

processed.write(filename=output_path)
