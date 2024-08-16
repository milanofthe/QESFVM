
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

#solve for f0
f0 = 100
omega = 2*np.pi*f0

#solve for the potential
Sim.solve(omega)

#compute the gradients
Sim.compute_E_J(omega)


# MAP RESULTS TO GRID FOR PLOTTING ================================================

(x_min, y_min), (x_max, y_max) = Sim.bounding_box

ar = (y_max - y_min)/(x_max - x_min)

nx = 100
ny = int(nx*ar)

grid_x = np.linspace(x_min, x_max, nx)
grid_y = np.linspace(y_min, y_max, ny)

grid = np.meshgrid(grid_x, grid_y)

#map potential to grid
Phi = Sim.map_to_grid(grid, "Phi_re") + 1j * Sim.map_to_grid(grid, "Phi_im")

#map electric field to grid
Ex = Sim.map_to_grid(grid, "E_x_re") + 1j * Sim.map_to_grid(grid, "E_x_im")
Ey = Sim.map_to_grid(grid, "E_y_re") + 1j * Sim.map_to_grid(grid, "E_y_im")



# PLOT RESULTS ====================================================================

#plot the results
fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 14), dpi=100, tight_layout=True)

(x_min, y_min), (x_max, y_max) = Sim.bounding_box
(l, t), (r, b) = Sim.bounding_box


#plot the mesh 
ax[0, 0].set_aspect(1)
ax[0, 0].set_title("mesh")

rectangles = []

for cell in Sim.QT.get_leafs():
    x, y = cell.center
    w, h = cell.size

    rectangles.append(Rectangle(xy=[x-w/2, y-h/2], 
                                width=w, 
                                height=h, 
                                ec="k", 
                                fc="none",
                                lw=0.7))

for material in Sim.materials:
    for cell in Sim.QT.get_leafs_inside_polygon(material.polygon):
        x, y = cell.center
        w, h = cell.size

        rectangles.append(Rectangle(xy=[x-w/2, y-h/2], 
                                    width=w, 
                                    height=h, 
                                    ec="k", 
                                    fc=material.color,
                                    lw=0.7))

ax[0, 0].add_collection(PatchCollection(rectangles, match_original=True))

for mat in Sim.materials:
    x, y = zip(*mat.polygon)
    ax[0, 0].fill(x, y, ec="k", fc="none", lw=2)
    
for prt in Sim.ports:
    if prt.p_src:
        ax[0, 0].scatter(*prt.p_src, marker="x", c="r", s=50, lw=2)
    if prt.p_snk:
        ax[0, 0].scatter(*prt.p_snk, marker=".", c="r", s=50, lw=2)

ax[0, 0].set_xlim(x_min, x_max)
ax[0, 0].set_ylim(y_min, y_max)

ax[0, 0].set_xlabel("x [m]")
ax[0, 0].set_ylabel("y [m]")


#plot the electric field (real part)
ax[0, 1].set_aspect(1)
ax[0, 1].set_title("E (log)")

C = abs(abs(E_x.real) + 1j* abs(E_y.real))
C = np.log10(np.where(C>1e-5, C, np.nan))

#streamplot
im = ax[0, 1].streamplot(grid_x, grid_y, E_x.real, E_y.real, color=C, density=3, cmap="viridis", zorder=0)
add_colorbar(im.lines, label="E (log) [V/m]")

for mat in Sim.materials:
    x, y = zip(*mat.polygon)
    ax[0, 1].fill(x, y, ec="k", fc="none", lw=2)

for prt in Sim.ports:
    if prt.p_src:
        ax[0, 1].scatter(*prt.p_src, marker="x", c="r", s=50, lw=2)
    if prt.p_snk:
        ax[0, 1].scatter(*prt.p_snk, marker=".", c="r", s=50, lw=2)

ax[0, 1].set_xlabel("x [m]")
ax[0, 1].set_ylabel("y [m]")

#set limits
ax[0, 1].set_ylim(y_min, y_max)
ax[0, 1].set_xlim(x_min, x_max)


#plot the electric potential (real part)
ax[1, 0].set_aspect(1)
ax[1, 0].set_title("Re(Phi)")

im = ax[1, 0].imshow(Phi.real, extent=[l, r, b, t])
add_colorbar(im, label="Re(Phi) [V]")


for mat in Sim.materials:
    x, y = zip(*mat.polygon)
    ax[1, 0].fill(x, y, ec="k", fc="none", lw=2)
    
for prt in Sim.ports:
    if prt.p_src:
        ax[1, 0].scatter(*prt.p_src, marker="x", c="r", s=50, lw=2)
    if prt.p_snk:
        ax[1, 0].scatter(*prt.p_snk, marker=".", c="r", s=50, lw=2)

ax[1, 0].set_xlim(x_min, x_max)
ax[1, 0].set_ylim(y_min, y_max)

ax[1, 0].set_xlabel("x [m]")
ax[1, 0].set_ylabel("y [m]")


#plot the electric potential (imaginary part)
ax[1, 1].set_aspect(1)
ax[1, 1].set_title("Im(Phi)")

im = ax[1, 1].imshow(Phi.imag, extent=[l, r, b, t])
add_colorbar(im, label="Im(Phi) [V]")

for mat in Sim.materials:
    x, y = zip(*mat.polygon)
    ax[1, 1].fill(x, y, ec="k", fc="none", lw=2)
    
for prt in Sim.ports:
    if prt.p_src:
        ax[1, 1].scatter(*prt.p_src, marker="x", c="r", s=50, lw=2)
    if prt.p_snk:
        ax[1, 1].scatter(*prt.p_snk, marker=".", c="r", s=50, lw=2)

ax[1, 1].set_xlim(x_min, x_max)
ax[1, 1].set_ylim(y_min, y_max)

ax[1, 1].set_xlabel("x [m]")
ax[1, 1].set_ylabel("y [m]")

plt.show()
