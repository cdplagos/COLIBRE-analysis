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

citation = "Kennicutt et al. (1998) [starbursts]"
bibcode = "1998ApJ...498..541K"
plot_as = "points"
name = (
    "Galaxy-averaged H2 Gas Surface Density vs Star " "Formation Rate Surface Density"
)

comment = f"Only including starbursts"

Sigma_H2 = []
Sigma_star = []
with open(f"../raw/Kennicutt1998_starbursts.txt", "r") as handle:
    for line in handle.readlines():
        if line[0] == "#":
            continue
        data = line.split()
        Sigma_H2.append(10.0 ** float(data[-3]))
        Sigma_star.append(10.0 ** float(data[-2]))

Sigma_H2 = unyt.unyt_array(Sigma_H2, "Msun/pc**2")
Sigma_star = unyt.unyt_array(Sigma_star, "Msun/yr/kpc**2")

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

output_path = f"../Kennicutt1998_starbursts.hdf5"

if os.path.exists(output_path):
    os.remove(output_path)

processed.write(filename=output_path)
