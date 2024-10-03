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

input_filename = f"../raw/data_schruba2011.txt"

processed = ObservationalData()

comment = "Based on the IRAM HERCULES survey."
citation = "Schruba et al. (2011)"
bibcode = "2011AJ....142...37S"
name = "azimuthally-averaged $\\Sigma_{\\rm HI + H_2}$ vs $\\Sigma_{\\rm SFR}$"
plot_as = "points"

(
    sigma_SFR,
    sigma_SFR_err,
    sigma_HI,
    sigma_HI_err,
    sigma_H2,
    sigma_H2_err,
) = np.genfromtxt(input_filename, unpack=True, usecols=(3, 4, 5, 6, 7, 8))

array_of_interest = np.arange(-1, 3, 0.25)
minimum_surface_density = 0.0  # The paper says 1 Msun/pc^2 is the limit in molecular gas, limit in Sigma_SFR is not yet reached at that surface density.
array_of_interest = array_of_interest[array_of_interest >= minimum_surface_density]
if array_of_interest[0] > minimum_surface_density:
    array_of_interest = np.append([minimum_surface_density], array_of_interest)

Obs_HI = (sigma_HI) / 1.36  # a factor of 1.36 to account for heavy elements
Obs_H2 = (sigma_H2) / 1.36  # a factor of 1.36 to account for heavy elements

Obs_Hneutral = Obs_HI + Obs_H2

Obs_SFR = sigma_SFR

mask_positive = (Obs_Hneutral > 0.0) & (Obs_SFR > 0.0)
Obs_Hneutral = Obs_Hneutral[mask_positive]
Obs_SFR = Obs_SFR[mask_positive]

binned_data = bin_data_general(
    np.log10(Obs_Hneutral), np.log10(Obs_SFR), array_of_interest
)

SigmaHneutral = unyt.unyt_array(10 ** binned_data[0], units="Msun/pc**2")

SigmaSFR = unyt.unyt_array(10 ** binned_data[1], units="Msun/yr/kpc**2")

SigmaSFR_err = unyt.unyt_array(
    [
        np.abs(10 ** (binned_data[1]) - 10 ** (binned_data[1] + binned_data[2])),
        np.abs(10 ** (binned_data[1] - binned_data[3]) - 10 ** (binned_data[1])),
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
    description="$\\Sigma_{\\rm HI + H_2}$",
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

output_path = f"../Schruba2011.hdf5"

if os.path.exists(output_path):
    os.remove(output_path)

processed.write(filename=output_path)
