
###################################################################################
##
##                      Example Simulation with QESFVM
##
##                            Milan Rother 2024
##
###################################################################################

# IMPORTS =========================================================================

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from qesfvm.simulation import Simulation
from qesfvm.materials import Material
from qesfvm.ports import Port
from qesfvm.utils import add_colorbar
    

# SETUP SIMULATION ================================================================

#geometric parameters
d_sub_1 = 300e-6
d_sub_2 = 300e-6

d_med_1 = 200e-6
d_med_2 = 50e-6

d_cell = 50e-6

d_mem = 15e-6

d_el = 500e-9
w_el = 100e-6
s_el = 150e-6

#simulation discretization
d_min, d_max = 1e-6, 5e-6

#electrical modeling of semipermeable membrane
sig_polyester = 1e-10 
sig_pbs = 1.64
eps_r_polyester = 3.5
eps_r_pbs = 1.0

hole_density = 6e5 * 1e4
hole_size = 3e-6
p = np.pi * (hole_size/2)**2 * hole_density # porosity

porous_polyester_sig_x = (1 - p) * p * sig_polyester * sig_pbs / ((1 - p) * sig_polyester + p * sig_pbs)
porous_polyester_sig_y = (1 - p) * sig_polyester + p * sig_pbs
porous_polyester_eps_r_x = (1 - p) * p * eps_r_polyester * eps_r_pbs / ((1 - p) * eps_r_polyester + p * eps_r_pbs)
porous_polyester_eps_r_y = (1 - p) * eps_r_polyester + p * eps_r_pbs

#electrical modelling of gold-electrolyte interface (Helmholtz model)
C_dl = 20e-6 * 10000 #double layer capacitance per m^2
d_dl = 10e-9         #double layer thickness

#cell monolayer parameters
epsilon_0 = 8.854187e-12
C_cell = 4e-2
TEER = 100

#simulation boundary
w_sim = 2*w_el + 2*s_el
d_sim = d_sub_1 + d_sub_2 + d_med_1 + d_med_2 + d_mem

#simulation bounding box
bounding_box = [[-w_sim/2, 0], [w_sim/2, d_sim]]

#electrical modeling of cell layer
TEERm2 = TEER / 10000   # <- normalize to square meter
eps_r_cells = C_cell * d_cell / epsilon_0
sig_cells  = d_cell / TEERm2 if TEER > 0.0 else 0.0


#define materials
materials = []

materials.append(Material("bottom_substrate_glass", 
                  polygon=[(-w_sim/2,0),
                           (w_sim/2,0),
                           (w_sim/2,d_sub_2), 
                           (-w_sim/2,d_sub_2)], 
                  eps_r=4.7, 
                  sig=0.0,   
                  d_min=d_el, 
                  color="lightgrey"))

materials.append(Material("top_substrate_glass", 
                  polygon=[(-w_sim/2,d_sub_2+d_med_2+d_mem+d_med_1),
                           (w_sim/2,d_sub_2+d_med_2+d_mem+d_med_1),
                           (w_sim/2,d_sub_2+d_med_2+d_mem+d_med_1+d_sub_1), 
                           (-w_sim/2,d_sub_2+d_med_2+d_mem+d_med_1+d_sub_1)], 
                  eps_r=4.7, 
                  sig=0.0,   
                  d_min=d_el,
                  color="lightgrey"))

materials.append(Material("porous_membrane_polyester", 
                  polygon=[(-w_sim/2,d_sub_2+d_med_2), 
                           (w_sim/2,d_sub_2+d_med_2), 
                           (w_sim/2,d_sub_2+d_med_2+d_mem), 
                           (-w_sim/2,d_sub_2+d_med_2+d_mem)], 
                  eps_r=(porous_polyester_eps_r_x, porous_polyester_eps_r_y), 
                  sig=(porous_polyester_sig_x, porous_polyester_sig_y), 
                  color="lightgreen"))

#only add cell layer if TEER is specified
if TEER > 0.0:
    materials.append(Material("cells", 
                      polygon=[(-w_sim/2,d_sub_2+d_med_2+d_mem), 
                               (w_sim/2,d_sub_2+d_med_2+d_mem), 
                               (w_sim/2,d_sub_2+d_med_2+d_mem+d_cell), 
                               (-w_sim/2,d_sub_2+d_med_2+d_mem+d_cell)], 
                     eps_r=eps_r_cells, 
                     sig=sig_cells,  
                     color="pink"))


#electrodes
M_el_1 = Material(f"bottom_left", 
                  polygon=[(-s_el/2-w_el,d_sub_2), 
                           (-s_el/2,d_sub_2),
                           (-s_el/2,d_sub_2+d_el), 
                           (-s_el/2-w_el,d_sub_2+d_el)], 
                  eps_r=1.0, 
                  sig=41.1e6,  
                  d_min=d_el/2,
                  color="khaki")
materials.append(M_el_1)

M_el_2 = Material(f"bottom_right", 
                  polygon=[(s_el/2+w_el,d_sub_2), 
                           (s_el/2,d_sub_2),
                           (s_el/2,d_sub_2+d_el), 
                           (s_el/2+w_el,d_sub_2+d_el)],
                  eps_r=1.0, 
                  sig=41.1e6,  
                  d_min=d_el/2,
                  color="khaki")
materials.append(M_el_2)

M_el_3 = Material(f"top_left", 
                  polygon=[(-s_el/2-w_el,d_sub_2+d_med_2+d_mem+d_med_1),
                           (-s_el/2,d_sub_2+d_med_2+d_mem+d_med_1),
                           (-s_el/2,d_sub_2+d_med_2+d_mem+d_med_1-d_el), 
                           (-s_el/2-w_el,d_sub_2+d_med_2+d_mem+d_med_1-d_el)], 
                  eps_r=1.0, 
                  sig=41.1e6,  
                  d_min=d_el/2,
                  color="khaki")
materials.append(M_el_3)

M_el_4 = Material(f"top_right", 
                  polygon=[(s_el/2+w_el,d_sub_2+d_med_2+d_mem+d_med_1),
                           (s_el/2,d_sub_2+d_med_2+d_mem+d_med_1),
                           (s_el/2,d_sub_2+d_med_2+d_mem+d_med_1-d_el), 
                           (s_el/2+w_el,d_sub_2+d_med_2+d_mem+d_med_1-d_el)], 
                  eps_r=1.0, 
                  sig=41.1e6,  
                  d_min=d_el/2,
                  color="khaki")
materials.append(M_el_4)


#background material (no polygon)
M_pbs = Material("PBS", eps_r=80.0, sig=1.64, color="lightblue")

#define port 
P1 = Port("port_1", 
          p_src=(-(w_el+s_el)/2, d_sub_2+d_med_2+d_mem+d_med_1-d_el/2), 
          p_snk=((w_el+s_el)/2,d_sub_2+d_el/2), 
          I_src=1.0)

P2 = Port("port_2", 
          p_src=((w_el+s_el)/2, d_sub_2+d_med_2+d_mem+d_med_1-d_el/2), 
          p_snk=(-(w_el+s_el)/2,d_sub_2+d_el/2), 
          I_src=1.0)

#turn off second port
P2.off()

#create subcell interfaces (electrode-electrolyte)
interfaces = [Interface("double_layer_1", material_a=M_pbs, material_b=M_el_1, cap=C_dl, delta=d_dl), 
              Interface("double_layer_2", material_a=M_pbs, material_b=M_el_2, cap=C_dl, delta=d_dl), 
              Interface("double_layer_3", material_a=M_pbs, material_b=M_el_3, cap=C_dl, delta=d_dl), 
              Interface("double_layer_4", material_a=M_pbs, material_b=M_el_4, cap=C_dl, delta=d_dl)]

#initialize simulation with materials and ports
Sim = Simulation(bounding_box=bounding_box, 
                 materials=materials, 
                 interfaces=interfaces,
                 ports=[P1, P2], 
                 d_min=d_min, 
                 d_max=d_max, 
                 boundary_condition_x_min=("periodic", None), 
                 boundary_condition_x_max=("periodic", None), 
                 boundary_condition_y_min=("neumann", None), 
                 boundary_condition_y_max=("neumann", None), 
                 background_material=M_pbs)


# SETUP SIMULATION ================================================================

#setup the simulation (precompute mesh, etc.)
Sim.setup()


# RUN SIMULATION ==================================================================

#frequency sweep of multiport impedance
omegas = 2 * np.pi* np.logspace(1, 6, 50)

#compute impedance matrix for frequencies
Z = np.array([Sim.solve_multiport(omega) for omega in omegas])


# PLOT RESULTS ====================================================================

fig, ax = plt.subplots(tight_layout=True, figsize=(6, 4), dpi=120)

ax.plot(freq, abs(Z[:,0,0]))
ax.plot(freq, abs(Z[:,1,0]))

ax.set_ylabel(r"impedance magnitude")
ax.set_xlabel("frequency [Hz]")

ax.set_yscale("log")
ax.set_xscale("log")
