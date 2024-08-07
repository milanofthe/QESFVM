###################################################################################
##
##                       Subcell Interface Definition
##
##                          Milan Rother 2023/24
##
###################################################################################


# IMPORTS =========================================================================

# no dependencies


# CLASSES =========================================================================

class Interface:

    def __init__(self, label="double_layer", material_a=None, material_b=None, delta=1e-9, cap=1e-6, color="none"):
        """
        managing capacitive sub cell interfaces between materials

        INPUTS : 
            label      : (str) name of the material interface
            material_a : ('Material' object) materials between which the interface is established
            material_b : ('Material' object) materials between which the interface is established
            delta      : (float) physical dimension of interface layer
            cap        : (float) interface capacitance defined per square meter [F/m^2]
            color      : (color) color of interface for visualization
        """

        self.label = label

        #materials that form interface
        self.material_a = material_a
        self.material_b = material_b

        #interface thickness and capacitance
        self.delta = delta
        self.cap = cap

        #for visualization
        self.color = color

        #tracking neighbouring cells
        self.pairs = set()


    def reset(self):
        self.pairs = set()

    def add(self, pair):
        self.pairs.add(pair)

    def is_interface(self, pair):
        return pair in self.pairs