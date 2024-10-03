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

citation = "Kennicutt et al. (1998) [spirals]"
bibcode = "1998ApJ...498..541K"
plot_as = "points"
name = (
    "Galaxy-averaged H2 Gas Surface Density vs Star " "Formation Rate Surface Density"
)

comment = f"Only including spirals"

data = np.loadtxt(
    f"../raw/Kennicutt1998_spirals.txt",
    usecols=(3, 5),
    dtype=[("Sigma_H2", np.float32), ("Sigma_star", np.float32)],
)

Sigma_H2 = unyt.unyt_array(10.0 ** data["Sigma_H2"], "Msun/pc**2")
Sigma_star = unyt.unyt_array(10.0 ** data["Sigma_star"], "Msun/yr/kpc**2")

processed.associate_x(
    Sigma_H2, scatter=None, comoving=False, description="H2 Surface density"
)
processed.associate_y(
    Sigma_star,
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

output_path = f"../Kennicutt1998_{type}.hdf5"

if os.path.exists(output_path):
    os.remove(output_path)

processed.write(filename=output_path)
