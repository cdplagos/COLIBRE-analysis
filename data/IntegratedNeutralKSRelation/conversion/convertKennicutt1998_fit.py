from velociraptor.observations.objects import ObservationalData

import unyt
import numpy as np
import os
import sys
import matplotlib.pyplot as plt


# Exec the master cosmology file passed as first argument
with open(sys.argv[1], "r") as handle:
    exec(handle.read())

processed = ObservationalData()

comment = "galaxy-averaged"
citation = "Kennicutt et al. (1998)"
bibcode = "1998ApJ...498..541K"
name = (
    "galaxy-averaged H2+HI Gas Surface Density vs Star Formation Rate Surface Density"
)
plot_as = "line"

# Reading the Kennicutt 1998 data

array_of_interest = np.arange(1, 3 + 0.25, 0.25)


def KS(sigma_g, n, A):
    return A * sigma_g ** n


Sigma_g = 10 ** array_of_interest
Sigma_star = KS(Sigma_g, 1.4, 1.515e-4)

Sigma_H2 = Sigma_g
Sigma_SFR = Sigma_star

SigmaH2 = unyt.unyt_array(Sigma_H2, units="Msun/pc**2")
SigmaSFR = unyt.unyt_array(Sigma_SFR, units="Msun/yr/kpc**2")

processed.associate_x(
    SigmaH2, scatter=None, comoving=False, description="H2+HI Surface density"
)
processed.associate_y(
    SigmaSFR,
    scatter=None,
    comoving=False,
    description="Star Formation Rate Surface Density",
)

processed.associate_citation(citation, bibcode)
processed.associate_name(name)
processed.associate_comment(comment)
processed.associate_redshift(0.0, 0.0, 0.0)
processed.associate_plot_as(plot_as)
processed.associate_cosmology(cosmology)

output_path = f"../Kennicutt1998_fit.hdf5"

if os.path.exists(output_path):
    os.remove(output_path)

processed.write(filename=output_path)
