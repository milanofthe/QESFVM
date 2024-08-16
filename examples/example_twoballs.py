
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

#circular electrode radius
r = 0.5
d = 0.5
k = 1.5

#simulation region
w = 6
h = 4

#bounding box
bounding_box = [[-w/2, -h/2], [w/2, h/2]]

#grid parameters
d_min, d_max = min(w, h)/160, min(w, h)/40

#circle as polygon
poly_circ = [(z.real, z.imag) for z in r * np.exp(2j * np.pi * np.linspace(0, 1, 33))]

#define materials
M1 = Material(label="ball_electrode_1", 
              polygon=poly_circ, 
              eps_r=1.0, 
              sig=58e6,
              d_min=r/20,
              color="gold")

M1.translate(dx=-r-d/2, dy=-d/2)

M2 = Material(label="ball_electrode_2", 
              polygon=poly_circ, 
              eps_r=1.0, 
              sig=58e6,
              d_min=r/20,
              color="gold")

M2.translate(dx=r+d/2, dy=d/2)

M3 = Material(label="dielectric", 
              polygon=[(-d/2, -k/2), (d/2, -k/2), (d/2, k/2), (-d/2, k/2)], 
              eps_r=6.5, 
              sig=0.0,
              d_min=r/10,
              color="lightgreen")


#ports
P1 = Port("P1", p_src=(-r-d/2, -d/2), p_snk=(r+d/2, d/2), I_src=None, V_src=1.0)
P2 = Port("P2", p_src=(-r-d/2, -d/2), p_snk=None, I_src=None, V_src=0.5)


#simulation
Sim = Simulation(bounding_box=bounding_box, 
                 materials=[M3, M1, M2], 
                 ports=[P1, P2], 
                 d_min=d_min, 
                 d_max=d_max,
                 boundary_condition_x_min=("periodic", None), 
                 boundary_condition_x_max=("periodic", None), 
                 boundary_condition_y_min=("neumann", None), 
                 boundary_condition_y_max=("neumann", None), 
                 refine_boundary_x_min=True,
                 refine_boundary_x_max=True,
                 refine_boundary_y_min=True,
                 refine_boundary_y_max=True)


# SETUP SIMULATION ================================================================

#setup the simulation (precompute mesh, etc.)
Sim.setup()


# RUN SIMULATION ==================================================================

#solve for f=10Hz
omega = 2*np.pi*10
Sim.solve(omega)
Sim.compute_E_J(omega)


# MAP RESULTS TO GRID FOR PLOTTING ================================================

(x_min, y_min), (x_max, y_max) = Sim.bounding_box

grid_x = np.linspace(x_min, x_max, 200)
grid_y = np.linspace(y_min, y_max, 100)

grid = np.meshgrid(grid_x, grid_y)

Phi = Sim.map_to_grid(grid, "Phi_re") + 1j * Sim.map_to_grid(grid, "Phi_im")
E_x = Sim.map_to_grid(grid, "E_x_re") + 1j * Sim.map_to_grid(grid, "E_x_im")
E_y = Sim.map_to_grid(grid, "E_y_re") + 1j * Sim.map_to_grid(grid, "E_y_im")


# PLOT RESULTS ====================================================================

#plot the results
fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(16, 9), dpi=100, tight_layout=True)

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
