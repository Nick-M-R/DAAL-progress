from shapely.geometry import Polygon
import numpy as np


class Airfoil:
    def __init__(self,
                 name = None, x = None, z = None):
        if name:
            self.name = name 
        else:
            self.name = "Airfoil"
        self.x = x
        self.z = z
            
        le_index = np.argmin(x) # split top and bottom surfaces
        self.x_top = np.flip(x[:le_index+1])
        self.z_top = np.flip(z[:le_index+1])
        self.x_bot = x[le_index:]
        self.z_bot = z[le_index:]

    def save(self, path):
        with open(path, "w") as file:
            file.write(self.name)
            file.writelines([f"\n{x} {z}" for (x, z) in zip(self.x, self.z)])
            file.write('\n')
    
    def is_valid(self):
        return Polygon(np.column_stack([self.x,self.z])).is_valid

    def min_radius(self): 
        dx = np.gradient(self.x)
        ddx = np.gradient(dx)
        
        dz = np.gradient(self.z)
        ddz = np.gradient(dz)
        
        r = (dx**2+dz**2)**1.5 / abs(dx*ddz - dz*ddx)
        
        return np.amin(r)
    
    def te_angle(self, units = "deg"): # can be "rad"
        theta = np.arctan(self.z_top[-2]/(1-self.x_top[-2])) - np.arctan(self.z_bot[-2]/(1-self.x_bot[-2]))
        
        if units == "deg":
            theta = theta*180/np.pi
        
        return theta
    
    def thickness(self, bounds = (0,1)):
        mask = ( self.x >= bounds[0] ) & ( self.x <= bounds[1] )
        
        x = self.x[mask]
        return np.interp(x, self.x_top, self.z_top) - np.interp(x, self.x_bot, self.z_bot)
    
    def camber(self):
        return (np.interp(self.x, self.x_top, self.z_top) + np.interp(self.x, self.x_bot, self.z_bot))/2
        
    def max_thickness(self):
        t = self.thickness()
        t_index = np.argmax(t)
        return t[t_index], self.x[t_index]
    
    def min_thickness(self, bounds = (0,1)):
        t = self.thickness(bounds = bounds)
        t_index = np.argmin(t)
        return t[t_index], t_index
    
    def max_camber(self):
        c = self.camber()
        c_index = np.argmax(c)
        return c[c_index], self.x[c_index]