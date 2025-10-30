#%%
from MOOFoil.basis import CST, CSTCamberThickness, plot_shapes
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import numpy as np
import os

# basis = CST(n_top = 16, n_bot = 16, le_mod = True, le_cont = False, finite_te=True)
bases = [
    CST(n_top = 8, n_bot = 8, le_mod = True, le_cont = True, finite_te=True),
    CSTCamberThickness(n_camber=8, n_thickness=8, le_mod = True, finite_te = True)
]

def fit(basis, x = None , z = None):
    t = np.linspace(1, -1, 301)
    x_interp = (0.5 - 0.5*np.cos(np.pi*np.abs(t)**1.5))
    A = basis.construct_A(x_interp)
    
    c, error = basis.fit(x,z)
    fig, ax = plt.subplot_mosaic(
        [["left", "upper right"],
         ["left", "lower right"]],
        figsize=[9,6], gridspec_kw={'height_ratios': [4,1], 'width_ratios': [3,1]}
    )
    for i,(y, color) in enumerate(zip((c*A+np.tile(np.linspace(2/len(c), 1.5+2/len(c), len(c)), (len(x_interp),1))).T, cm.hsv(np.linspace(0, 1, len(c))))):
        ax["upper right"].plot(x_interp, y, color = color)
        ax["upper right"].text(1.02, y[0], f"{i}")
        
    A_error = basis.construct_A(x)

    z_error = np.abs(A_error@c - z)
    z_error[:np.argmin(x)] = -z_error[:np.argmin(x)] 
    
    ax["upper right"].fill_between(x_interp,A@c, alpha = 0.2)
    ax["upper right"].plot(x_interp,A@c, 'k')
    ax["upper right"].set_yticks([])
    ax["upper right"].axis("equal")
    ax["upper right"].set_xlim([-0.2,1.2])
    ax["upper right"].set_ylim([-0.05,1.7])
    
    ax["left"].plot(x_interp,A@c, 'k')
    ax["left"].plot(x,z, '--or', linewidth = 1)

    ax["lower right"].plot(x,z_error,label = "Fit error")
    ax["lower right"].legend()
    ax["lower right"].set_xlabel("x/c")
    ax["lower right"].get_yaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: f"{np.abs(x):.1e}"))
    
    
    fig.tight_layout()
    return error

if __name__ == "__main__":
    data = "basis_dat"
    hist_data = {basis.__class__:[] for basis in bases}
    for i, file in enumerate([f"{data}/{f}" for f in os.listdir(data)]):
        for basis in bases:
            error = fit(basis, *np.loadtxt(file, skiprows = 1, unpack = True))
            log10error = np.log10(error)
            # if error > 0.005:
            print()
            print(i, basis.__class__,file.split('/')[-1],error/0.005)
            #     plt.close()
            plt.suptitle(f"{file.split('.')[0]} | {basis.__class__}")
            hist_data[basis.__class__].append(log10error)
            
    # for basis in bases:
    #     plt.hist(hist_data[basis.__class__], bins = 50, alpha=0.5, label = basis.__class__)
    # plt.legend()
    # plt.title("Comparison of CST to CST Camber Thickness | + LE modification")
    # plt.xlabel("log(rms)")
    # plt.ylabel("Airfoils")
    # e387
    
    # error = fit(*np.loadtxt("basis_dat/e387.dat", skiprows = 1, unpack = True))
    # print(np.log10(error))
    # plt.show()

    # for basis in bases:
    #     plot_shapes(basis)
    plt.show()
        
# %%
