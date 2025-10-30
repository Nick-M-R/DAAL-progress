"""
MOOFoil.basis

A collection of airfoil parameterization methods.

A basis class includes the following:

1. A method to fit airfoils      | fit(x,z, **kwargs) 
    Inputs to fit must be in Selig format. Returns coefficients and measured error.
2. A method to generate airfoils | generate(x, k, **kwargs) 
    Returns x, z values in Selig format.
    
""" 

# print("Imported MOOFoil.basis")
import numpy as np
from scipy.special import binom
from scipy.linalg import lstsq
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from os import listdir

class CST:
    def __init__(self, n_top, n_bot, shape_function = "Bernstein", le_mod = True, le_cont = True, finite_te = True,
        lower_bound = -2, upper_bound = 2):
        match shape_function:
            case "Bernstein":
                self.shape_function = self.bernstein
            case _:
                self.shape_function = self.bernstein
                
        if n_top != n_bot and le_mod:
            le_mod = 2
        self.le_mod = le_mod
        self.le_cont = le_cont
        self.finite_te = finite_te
        
        self.n_top = n_top
        self.n_bot = n_bot
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
        self.n_var = n_top + n_bot + sum([le_mod, finite_te]) - sum([le_cont])
        
    def construct_A(self, x):
        le_index = np.argmin(x)
        
        x_top = x[:le_index+1]
        x_bot = x[le_index+1:]
        
        n_x_top = len(x_top)
        n_x_bot = len(x_bot)
        
        A = np.zeros((len(x), self.n_top + self.n_bot - self.le_cont))
        # get the basic basis functions for CST
        A_top = self.class_function(x_top, N1 = 0.5, N2 = 1.0) * self.shape_function(x_top, self.n_top)
        A_bot = -self.class_function(x_bot, N1 = 0.5, N2 = 1.0) * self.shape_function(x_bot, self.n_bot)
        A[:n_x_top, :self.n_top] = A_top

        if self.le_cont:
            A[-n_x_bot:, 0] = A_bot[:, 0]
            A[-n_x_bot:, self.n_top:self.n_top+self.n_bot-1] = A_bot[:,1:]
        else:
            A[-n_x_bot:, self.n_top:self.n_top+self.n_bot] = A_bot
        
        # collect extra terms
        A_extra = []
        
        # add the additional LE term(s)
        if self.le_mod == 2:
            A_extra.append(x*(1-x)**(self.n_top-0.5))
            A_extra.append(x*(1-x)**(self.n_bot-0.5))
            
        else:
            A_extra.append(x*(1-x)**(self.n_top-0.5))
            
        # get the trailing edge term if needed (ALWAYS LAST)
        if self.finite_te:
            A_extra.append(np.concatenate([x_top, -x_bot])/2)
            
            
        A = np.hstack((A, np.array(A_extra).T))

        return A
        
    def fit(self, x, z, interp = False):
        
        z = z[:] + 0 # wacky reference behavior
        x = x[:] + 0 
        if interp:
            t = np.linspace(0, 1, len(x))
            t_interp = np.linspace(0, 1, 4 * len(x))
            x = np.interp(t_interp, t, x)
            z = np.interp(t_interp, t, z)
            
        le_index = np.argmin(x)
        
        x_top = x[:le_index+1]
        x_bot = x[le_index+1:]
        
        x_min = x[le_index]
        x_max = np.amax(x)
        
        # normalize if necessary 
        if any((x_min < 0, x_max > 1)):
            x = np.interp(x, [np.amin((0, x_min)), x_max], [0,1])
            z = z/(x_max-x_min)
        
        c_thickness = z[0] - z[-1]
        z[:le_index+1] = z[:le_index+1] - z[0]*x_top
        z[le_index+1:] =  z[le_index+1:] - z[-1]*x_bot
        
        A = self.construct_A(x)
        if self.finite_te:
            A = A[:,:-1]
        
        # lamb = 1e-6
        # U,S,Vt = np.linalg.svd(A, full_matrices=False)
        # c = Vt.T@((U.T@z)*(S/(S**2+lamb)))
        # error = np.linalg.norm(A@c-z)
        
        # AA = A.T @ A
        # bA = z @ A
        # D, U = np.linalg.eigh(AA)
        # Ap = (U * np.sqrt(D)).T
        # bp = bA @ U / np.sqrt(D)
        # c = np.linalg.lstsq(Ap, bp, rcond=None)[0]
        # c = lstsq(Ap, bp, lapack_driver='gelsy')[0]
        
        fit = Ridge(alpha = 1e-8).fit(A,z)
        c = fit.coef_
        error = np.linalg.norm(A@c-z)
        
        if self.finite_te:
            c = np.concatenate([c, [c_thickness]])
        # plt.imshow(A)
        # plt.show()
        return c, error 
            
    def generate(self, x, c):
        z = self.construct_A(x)@c
        return x, z
    
    ## CST CLASS FUNCTION ##
    def class_function(self, x, N1 = 0.5, N2 = 1.0):
        x = np.vstack(x)
        return (x)**N1 * (1-x)**N2
    
    ## CST POLYNOMIALS ##
    def bernstein(self, x, n_var):
        k = np.arange(n_var)
        n = n_var-1
        x = np.vstack(x)
        return binom(n,k) * np.power(x, k) * np.power((1-x), n-k)

class CSTCamberThickness:
    def __init__(self, n_camber, n_thickness, shape_function = "Bernstein", le_mod = True, finite_te = True,
        lower_bound = -2, upper_bound = 2):
        match shape_function:
            case "Bernstein":
                self.shape_function = self.bernstein
            case _:
                self.shape_function = self.bernstein
                
        self.n_camber = n_camber
        self.n_thickness = n_thickness
        self.le_mod = le_mod
        self.finite_te = finite_te
        
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
        self.n_var = n_camber + n_thickness + 1*finite_te + 1*le_mod
        
    def construct_A(self, x):
        # discretize the top and bottom surfaces
        le_index = np.argmin(x)
        x_top = x[:le_index+1]
        x_bot = x[le_index+1:]

        A_top = []
        A_bot = []
        
        # camber contribution using parabola shape function
        A_top.append(self.class_function(x_top, N1 = 1.0, N2 = 1.0)*self.shape_function(x_top, self.n_camber))
        A_bot.append(self.class_function(x_bot, N1 = 1.0, N2 = 1.0)*self.shape_function(x_bot, self.n_camber))

        # thickness contribution
        A_top.append(self.class_function(x_top, N1 = 0.5, N2 = 1.0) * self.shape_function(x_top, self.n_thickness))
        A_bot.append(-self.class_function(x_bot, N1 = 0.5, N2 = 1.0) * self.shape_function(x_bot, self.n_thickness))

        # add the additional LE term(s)
        if self.le_mod:
            # similar results to kulfan LE mod, node is infront of first camber node
            A_top.append(((x_top**0.5)*(1-x_top)**(self.n_camber+0.5))[np.newaxis].T)
            A_bot.append(((x_bot**0.5)*(1-x_bot)**(self.n_camber+0.5))[np.newaxis].T)        
           
        # get the trailing edge term if needed (ALWAYS LAST)
        if self.finite_te:
            A_top.append(x_top[np.newaxis].T/2)
            A_bot.append(-x_bot[np.newaxis].T/2)
           
        A = np.block([A_top,A_bot])

        return A

    def fit(self, x, z, interp = False):
        
        z = z[:] + 0 # wacky reference behavior
        x = x[:] + 0 
        if interp:
            t = np.linspace(0, 1, len(x))
            t_interp = np.linspace(0, 1, 4 * len(x))
            x = np.interp(t_interp, t, x)
            z = np.interp(t_interp, t, z)
            
        le_index = np.argmin(x)
        
        x_top = x[:le_index+1]
        x_bot = x[le_index+1:]
        
        x_min = x[le_index]
        x_max = np.amax(x)
        
        # normalize if necessary 
        if any((x_min < 0, x_max > 1)):
            x = np.interp(x, [np.amin((0, x_min)), x_max], [0,1])
            z = z/(x_max-x_min)
        
        A = self.construct_A(x)
        
        fit = Ridge(alpha = 0).fit(A,z)
        c = fit.coef_
        error = np.linalg.norm(A@c-z)
        
        return c, error 
            
    def generate(self, x, c):
        z = self.construct_A(x)@c
        return x, z
    
    ## CST CLASS FUNCTION ##
    def class_function(self, x, N1 = 0.5, N2 = 1.0):
        x = np.vstack(x)
        return (x)**N1 * (1-x)**N2
    
    ## CST POLYNOMIALS ##
    def bernstein(self, x, n_var):
        k = np.arange(n_var)
        n = n_var-1
        x = np.vstack(x)
        return binom(n,k) * np.power(x, k) * np.power((1-x), n-k)

class SVD:
    def __init__(self, dat_folder, n_modes = 5):
        airfoils = listdir(dat_folder)
        x = np.loadtxt(f"{dat_folder}/{airfoils[0]}", usecols = 0, skiprows = 1)
        
        A = np.array([
            np.loadtxt(f"{dat_folder}/{af}", usecols = 1, skiprows = 1) for af in airfoils
        ]).T

        U, S, Vh = np.linalg.svd(A)
        
        self.modes = U[:,:n_modes]
        self.n_var = n_modes
        
    def fit(self, x, z):
        # normalize if necessary 
        c, error = np.linalg.lstsq(self.modes, z, rcond=None)[:2]

        return c, error
            
    def generate(self, x, c):
        return x, self.modes@c

def plot_shapes(basis):
    
    t = np.linspace(1, -1, 301)
    x = (0.5 - 0.5*np.cos(np.pi*np.abs(t)**1.5))

    A = basis.construct_A(x)
    c = 50*np.ones(basis.n_var)
    
    plt.figure()
    for i,(y, color) in enumerate(zip((c*A+np.tile(np.linspace(2/len(c),2/len(c), len(c)), (len(x),1))).T, cm.hsv(np.linspace(0, 1, len(c))))):
        plt.plot(x, y, color = color)
        plt.text(1.02, y[0], f"{i}")
    

if __name__ == "__main__":
    basis = CSTCamberThickness(n_camber=8, n_thickness=8, le_mod = False, finite_te = False,)
    t = np.linspace(1, -1, 301)
    x = (0.5 - 0.5*np.cos(np.pi*t))
    A = basis.construct_A(x)
    x, z = np.loadtxt()
    # import matplotlib.pyplot as plt
    # methods = [CST]
    
    
    # dat_folders = [
    #     "/home/canativi/Documents/Python/MOOFoil/tests/basis_dat",
    #     "/home/canativi/Documents/Python/MOOFoil/examples/safe/May/gen0",
    # ]
    
    # for method in methods:
    #     par = method(n_top = 16, n_bot = 8)
            
    #     ## debug data ##
    #     max_c = [0, None]
    #     min_c = [0, None]
    #     max_error = [0, None]
    #     print(f"{'Airfoil':<15.15} {'Max Coeff':<15.15} {'Min Coeff':<15.15} {'Fit Error':<15.15}")
    #     print("=============================================================")
    #     for fold in dat_folders:
    #         for af in listdir(fold):
    #             x, z = np.loadtxt(f"{fold}/{af}", unpack = True, skiprows = 1)
    #             # print(af)
    #             ## FITTING ##
    #             c, error = par.fit(x, z)
                
    #             if np.amin(c) > min_c[0]:
    #                 min_c = [np.amin(c), af]
    #             if np.amax(c) > max_c[0]:
    #                 max_c = [np.amax(c), af]
    #             if error > max_error[0]:
    #                 max_error = [error, af]
    #             print(f"{af:<15.15} {np.amax(c):<15.3e} {np.amin(c):<15.3e}{error:<15.3e}")
                
    #             # GENERATING
    #             t = np.linspace(1, -1, 601)
    #             x_interp = (0.5 - 0.5*np.cos(np.pi*t))
    #             x_c, z_c = par.generate(x_interp, c)
    #             plt.plot(x, z, label = "Original")
    #             plt.plot(x_c, z_c, label = "CST")
    #             plt.legend()
    #             plt.show()
            