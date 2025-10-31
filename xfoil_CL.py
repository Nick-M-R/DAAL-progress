import numpy as np
import os
import sys
from MOOFoil.basis import CSTCamberThickness
from pyXFOIL import XFOILCase
import matplotlib as mpl

def run_xfoil(x, z, CL_pair, Re, Ma, cpu_path, compute_CL_max=True):
    mpl.use('agg')
    try:
        CL_list = np.linspace(CL_pair[0], CL_pair[1], 11)
        xfoil_case = XFOILCase(x = x, z = z,
                               Re = Re, Ma = Ma, alpha = [], CL = CL_list,
                               Tu_pct = 0.05, xtr_top = 1, xtr_bot = 1,
                               clean = True,
                               run_path = f"{cpu_path}/XFOIL",
                               xfoil_path = "xfoil",
                               case_name = "0",
                               verbose = True, timeout = 25, iter = 75)

        xfoil_case.run()
        xfoil_polar = xfoil_case.get_polar()

        # Ensure both requested CL values are present in the polar
        for cl in CL_pair:
            assert cl in xfoil_polar["CL"]

        results = [{ k:v[i] for (k,v) in zip(xfoil_polar.keys(), xfoil_polar.values()) } for i in [0, -1]]

        if compute_CL_max:
            xfoil_case = XFOILCase(x = x, z = z,
                                   Re = Re, Ma = Ma, alpha = [], CL = [CL_pair[1]],
                                   Tu_pct = 0.05, xtr_top = 1, xtr_bot = 1,
                                   clean = True,
                                   run_path = f"{cpu_path}/XFOIL",
                                   xfoil_path = "xfoil",
                                   case_name = "1",
                                   verbose = False, timeout = 30)
            results.append({"CL_max": xfoil_case.CL_max()[0]})

        return results
    except Exception as e:
        print(f"XFOIL failed: {e}")
        return 0



if __name__ == "__main__":
    # Airfoil geometry file (required)
    if len(sys.argv) < 2:
        print("Usage: python run_CL_xf.py <airfoil_dat_file>")
        raise SystemExit(1)

    file = sys.argv[1]

    # Fixed flow conditions
    CL_pair = (0.3, 0.7)
    Re = 5.0e5
    Ma = 0.03
    cpu_path = "."

    os.makedirs(cpu_path, exist_ok=True)

    # Fit CST and generate a smooth geometry
    basis = CSTCamberThickness(n_camber = 8, n_thickness = 8,
                               le_mod = True, finite_te = True,
                               lower_bound = -2, upper_bound = 2)

    # Load coordinates
    x_data, z_data = np.loadtxt(file, skiprows=1, unpack=True)
    coeffs, _ = basis.fit(x_data, z_data)

    t_interp = np.linspace(1, -1, 301)
    x_interp = 0.5 - 0.5 * np.cos(np.pi * t_interp)
    x, z = basis.generate(x_interp, coeffs)

    # Run XFOIL for the two CL targets (0.3 and 0.7)
    result = run_xfoil(x, z, CL_pair, Re, Ma, cpu_path, compute_CL_max=True)
    if result == 0:
        raise SystemExit(1)
