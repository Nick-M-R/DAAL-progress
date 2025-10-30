#%%
from MOOFoil.basis import CST, CSTCamberThickness, plot_shapes
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import numpy as np
import os

basis = CSTCamberThickness(n_camber=8, n_thickness=8, le_mod = True, finite_te = True)

def fit_and_thicken(basis, x = None , z = None):
    t = np.linspace(1, -1, 301)
    x_interp = (0.5 - 0.5*np.cos(np.pi*np.abs(t)**1.5))
    A = basis.construct_A(x_interp)
    
    c, error = basis.fit(x,z)
    plt.figure()
    plt.plot(x_interp,A@c, label = "original")
    
    c[basis.n_camber:(basis.n_camber+basis.n_thickness)] = 2*c[basis.n_camber:(basis.n_camber+basis.n_thickness)]
    plt.plot(x_interp,A@c, label = "scaled")
    plt.title("DAE11")
    plt.legend()
    return error

if __name__ == "__main__":
    af_file = "basis_dat/DAE11.dat"
    fit_and_thicken(basis,*np.loadtxt(af_file, skiprows = 1, unpack = True))
    plt.show()

# %%
