###################################################################################
##
##                          Material Definitions
##
##                          Milan Rother 2023/24
##
###################################################################################


# IMPORTS =========================================================================

import numpy as np



# CLASSES =========================================================================

class Material:

    def __init__(self, label="air", polygon=[], eps_r=1.0, sig=0.0, color="none", d_min=None):
        """
        initialize the material

        INPUTS : 
            label    : (str) name of the material
            polygons : (list of list of points) polygon or list of polygons defining the material shape
            eps_r    : (scalar or tuple) anisotropic dielectric
            sig      : (scalar or tuple) anisotropic conductivity
            color    : (str) color for visualization (optional)
            d_min    : (float) minimum grid size specific for material
        """

        self.label = label
        self.polygon = polygon
        self.color = color

        #compute minimum bounding box of material
        if len(polygon) > 0:
            x, y = zip(*polygon)
            d_x, d_y = max(x)-min(x), max(y)-min(y)
            d_min = d_min if d_min else min(d_x, d_y)

        self.d_min = d_min

        #handle anisotropic epsilon
        if isinstance(eps_r, (tuple, list)):
            self.eps_r_x, self.eps_r_y = eps_r 
        else:
            self.eps_r_x = self.eps_r_y = eps_r

        #handle anisotropic sigma
        if isinstance(sig, (tuple, list)):
            self.sig_x, self.sig_y = sig 
        else:
            self.sig_x = self.sig_y = sig


        #initialize cell indices that belong to material
        self.indices = []


    def set_indices(self, indices):
        """
        set the cell indices that belong to the material
        """
        self.indices = indices



    def translate(self, dx=0.0, dy=0.0):
        """
        perform linear translation of material polygon
        """
        self.polygon = [(x+dx, y+dy) for x, y in self.polygon]


    def rotate(self, angle=0.0, deg=True):
        """
        perform rotation of material polygon with respect to origin
        """
        _angle = np.pi/180*angle if deg else angle
        def _rotate(point, angle):
            x, y = point
            _x = x * np.cos(angle) - y * np.sin(angle)
            _y = x * np.sin(angle) + y * np.cos(angle)
            return _x, _y
        self.polygon = [_rotate(p, _angle) for p in self.polygon]
        

