###################################################################################
##
##                           Port Definitions
##
##                         Milan Rother 2023/24
##
###################################################################################


# IMPORTS =========================================================================

# no dependencies


# CLASSES =========================================================================

class Port:

    def __init__(self, label="P", p_src=None, p_snk=None, I_src=None, V_src=None, active=True):
        """
        Lumped ideal hybrid port in the x-y-plane. 
        The port is defined as two points 'p_src' and 'p_snk' where 
        a current 'I_src' is injected (source -> sink). 
        
        The port can also retrieve the voltage 'V' between the two points 
        after the simulation has finished.

        Alternatively, a voltage 'V_src' can be set between the two points
        as a potential difference.

        INPUTS : 
            label  : (str) name of the source
            p_src  : (tuple) point in x-y-plane for current injection (source)
            p_snk  : (tuple) point in x-y-plane for current injection (sink)
            I_src  : (float) Amplitude of the current source [A]
            V_src  : (float) Amplitude of the voltage source [V]
            active : (bool) activate port (source term), default True
        """

        self.label = label

        #port points
        self.p_src = p_src
        self.p_snk = p_snk

        #port excitation values
        self.I_src = I_src
        self.V_src = V_src

        #set port voltage
        self.V = 0.0

        #flag for setting port active
        self.active = active

        #keeping track in which cell the source and sink are
        self.src_cell_idx = None
        self.snk_cell_idx = None

    def set_src_cell_idx(self, idx):
        self.src_cell_idx = idx

    def set_snk_cell_idx(self, idx):
        self.snk_cell_idx = idx

    def on(self):
        self.active = True

    def off(self):
        self.active = False
