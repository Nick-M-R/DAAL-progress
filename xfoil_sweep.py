"""Run an XFOIL sweep for a given airfoil.

Usage:
    python xfoil_sweep.py path/to/airfoil.dat

Notes:
    - Input file is expected to have two columns (x, z) with a header line.
    - If the input is not sampled at 301 points, it is re-fit with CST and
      re-sampled to 301 points.
"""

from pyXFOIL import XFOILCase
import numpy as np
import sys
import os
from MOOFoil.basis import CSTCamberThickness

Ma = 0.03
Re = 5e5

def get_basis(n=301):
    """Create a CST camber/thickness basis and cosine-spaced x-grid.

    Args:
        n: Number of x samples to generate.

    Returns:
        (basis, x_interp): CST basis object and 1D array of x locations.
    """
    basis = CSTCamberThickness(n_camber = 8, n_thickness = 8,
                           le_mod = True, finite_te = True,
                           lower_bound = -2, upper_bound = 2)
    t_interp = np.linspace(1, -1, n)
    x_interp = 0.5 - 0.5 * np.cos(np.pi * t_interp)
    return basis, x_interp

def run_xfoil(x, z, alpha, cpu_path):
    """Run XFOIL for a set of angles of attack.

    Args:
        x: 1D numpy array of x-coordinates.
        z: 1D numpy array of z-coordinates (airfoil surface, TE to TE order).
        alpha: Iterable of angles of attack (degrees).
        cpu_path: Directory to place XFOIL run outputs.

    Returns:
        XFOIL polar dict on success, or 0 on failure.
    """

    try:
        xfoil_case = XFOILCase(x = x, z = z,
          Re = Re, Ma = Ma, alpha = alpha, CL = [],
          Tu_pct = 0.05, xtr_top = 1, xtr_bot = 1,
          clean = True,
          run_path = f"{cpu_path}/XFOIL",
          xfoil_path = "xfoil",
          case_name = "0",
          verbose = True, timeout = 25, iter = 75)

        xfoil_case.run()
        xfoil_polar = xfoil_case.get_polar()

        return xfoil_polar

    except:
        return 0


if __name__ == "__main__":
    args = sys.argv

    # Determine input file: default to 'af.dat', override with argument if provided and exists
    filename = "af.dat"
    if len(args) > 1 and os.path.isfile(args[-1]):
        filename = args[-1]
    if not os.path.isfile(filename):
        raise FileNotFoundError("No input file provided and default 'af.dat' not found")

    alpha = np.arange(-4, 16, 1)
    x, z = np.loadtxt(filename, skiprows=1, unpack=True)

    if len(x) != 301 or len(z) != 301:
        # Fit CST and re-sample to 301 points
        basis, x_interp = get_basis(n=301)
        coeffs, _ = basis.fit(x, z)
        x, z = basis.generate(x_interp, coeffs)

    xfoil_polar = run_xfoil(x, z, alpha, ".")

    print(xfoil_polar)