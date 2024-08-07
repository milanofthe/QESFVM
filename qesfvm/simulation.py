

###################################################################################
##
##                  Quasi Electrostatic FVM Simulation Class
##
##                            Milan Rother 2023/24
##
###################################################################################


# IMPORTS =========================================================================

import numpy as np

import matplotlib.pyplot as plt

from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix#, csr_matrix, bmat, diags

from matplotlib import pyplot as plt

import logging

# from qesfvm.quadtree import QuadTree
from qesfvm.quadtree import QuadTree
from qesfvm.materials import Material
from qesfvm.utils import (
    performance_enumerate,
    segments_from_poly,
    timer, 
    qt_rl
    )





# SIMULATION CLASS ================================================================

class Simulation:

    """
    A Finite Volume Method for quasi-Electrostatic simulations in 2D 
    that utilizes a quadtree datastructure for adaptive mesh refinement.

    INUPUTS : 
        bounding_box             : (list) list of two points that define simulation space corners
        materials                : (list of 'Material' objects) material model definitions for simulation
        interfaces               : (list of 'Interface' objects) special material interface definitions
        ports                    : (list of 'Port' objects) port definitions for simulation
        d_min                    : (float) minimum grid size / size of smallest cell
        d_max                    : (float) maximum grid size / size of largest cell
        boundary_condition_x_min : (None -> neumann, float -> dirichlet) boundary condition left
        boundary_condition_x_max : (None -> neumann, float -> dirichlet) boundary condition right 
        boundary_condition_y_min : (None -> neumann, float -> dirichlet) boundary condition bottom 
        boundary_condition_y_max : (None -> neumann, float -> dirichlet) boundary condition top 
        refine_boundary_x_min    : (bool) mesh refinement at left boundary
        refine_boundary_x_max    : (bool) mesh refinement at right boundary
        refine_boundary_y_min    : (bool) mesh refinement at bottom boundary
        refine_boundary_y_max    : (bool) mesh refinement at top boundary
        background_material      : ('Material' object) defines material model for background
        log                      : (bool) logging mode with logging
    """ 

    def __init__(self, 
                 bounding_box=[[0,0],[1e-3,1e-3]], 
                 materials=[], 
                 ports=[], 
                 interfaces=[],
                 d_min=None, 
                 d_max=None, 
                 boundary_condition_x_min=("neumann", None), 
                 boundary_condition_x_max=("neumann", None), 
                 boundary_condition_y_min=("neumann", None), 
                 boundary_condition_y_max=("neumann", None), 
                 refine_boundary_x_min=False,
                 refine_boundary_x_max=False,
                 refine_boundary_y_min=False,
                 refine_boundary_y_max=False,
                 background_material=None, 
                 log=True):

        
        # Simulation boundary
        self.bounding_box = bounding_box

        # Background material (if None -> freespace)
        if background_material is None:
            self.background_material = Material("background", eps_r=1.0, sig=0.0, color="none")
        else:
            self.background_material = background_material

        # Grid sizes for refinement
        self.d_min = d_min
        self.d_max = d_max

        # Boundary conditions
        if isinstance(boundary_condition_x_min, str): 
            self.boundary_condition_x_min = (boundary_condition_x_min, 0.0)
        else: 
            self.boundary_condition_x_min = boundary_condition_x_min

        if isinstance(boundary_condition_x_max, str): 
            self.boundary_condition_x_max = (boundary_condition_x_max, 0.0)
        else: 
            self.boundary_condition_x_max = boundary_condition_x_max

        if isinstance(boundary_condition_y_min, str): 
            self.boundary_condition_y_min = (boundary_condition_y_min, 0.0)
        else: 
            self.boundary_condition_y_min = boundary_condition_y_min

        if isinstance(boundary_condition_y_max, str): 
            self.boundary_condition_y_max = (boundary_condition_y_max, 0.0)
        else: 
            self.boundary_condition_y_max = boundary_condition_y_max

        # Boundary refinement flags
        self.refine_boundary_x_min = refine_boundary_x_min
        self.refine_boundary_x_max = refine_boundary_x_max
        self.refine_boundary_y_min = refine_boundary_y_min
        self.refine_boundary_y_max = refine_boundary_y_max

        # Logging mode flag
        self.log = log

        # Natural constants
        self.eps_0 = 8.854187e-12

        # Initialize materials
        self.materials = materials
        self.ports = ports
        self.interfaces = interfaces





    def logger(self, message):
        if self.log:
            logging.info(message)

    def _setup_logging(self):

        if self.log:
            #if a filename for logging is specified
            filename = self.log if isinstance(self.log, str) else None

            #initialize the logging
            logging.basicConfig(level=logging.INFO,  
                                format="%(asctime)s - %(levelname)s - %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S",
                                filename=filename,  
                                filemode="w")

            logging.info("LOGGING ENABLED")











    def _setup_mesh(self):

        self.logger("MESH SETUP")

        #unpack bounding box
        (x1, y1), (x2, y2) = self.bounding_box

        #quadtree initialization with initial splits of quadtree root cell
        self.QT = QuadTree(bounding_box=self.bounding_box, 
                           n_initial_x=int(np.ceil(abs(x1-x2)/self.d_max)),
                           n_initial_y=int(np.ceil(abs(y1-y2)/self.d_max)))

        self.logger(f"progress - QuadTree -> initialized {len(self.QT)} cells")

        #compute number of global refinement levels
        global_refinement_levels = qt_rl(self.d_max, self.d_min)


        #collect segments for refinement from materials
        relevant_segments = {i:[] for i in range(global_refinement_levels)} 

        #iterate all materials 
        for mat in self.materials:

            #specific refinement for material
            if mat.d_min:
                m_rl = qt_rl(self.d_max, min(self.d_min, mat.d_min))
                material_refinement_levels = max(m_rl, global_refinement_levels)
            else:
                material_refinement_levels = global_refinement_levels

            #collect relevant segments for refinement
            for seg in segments_from_poly(mat.polygon):
                xs, ys = zip(*seg)
                tol = 1e-12
                if (sum([abs(x-x1) for x in xs]) < tol or 
                    sum([abs(x-x2) for x in xs]) < tol or 
                    sum([abs(y-y1) for y in ys]) < tol or 
                    sum([abs(y-y2) for y in ys]) < tol):
                    continue
                for i in range(material_refinement_levels):
                    if i not in relevant_segments:
                        relevant_segments[i] = [seg]
                    else:
                        relevant_segments[i].append(seg)

        #refine quadtree
        for level, segments in relevant_segments.items():

            #minimum cell size of previous refinement step
            d_min = min(min(cell.size) for cell in self.QT.get_leafs())
            if segments:
                # introduce tolerance to enforce equal sized cells at interfaces
                self.QT.refine_edge(segments,    
                                    min_size=d_min/2,
                                    tol=d_min)  
            
            #boundary refinement of quadtree
            self.QT.refine_boundary(x_min=self.refine_boundary_x_min,
                                    x_max=self.refine_boundary_x_max,
                                    y_min=self.refine_boundary_y_min,
                                    y_max=self.refine_boundary_y_max,  
                                    min_size=d_min/2)

            self.logger(f"progress - refinement level {level} -> {len(self.QT)} cells")

        #balance quadtree mesh after refinement
        self.QT.balance()

        self.logger(f"progress - balanced -> {len(self.QT)} cells")

        


















    def _setup_materials(self):

        self.logger("MATERIAL SETUP")

        #get all the leafs
        leafs = self.QT.get_leafs()

        #setup cell indices
        for i, cell in enumerate(self.QT.get_leafs()):
            cell.set("idx", i)

        #first set background material everywhere
        background_indices = []
        for cell in self.QT.get_leafs():
            background_indices.append(int(cell.get("idx")))

            cell.set("eps_r_x", self.background_material.eps_r_x)
            cell.set("eps_r_y", self.background_material.eps_r_y)
            cell.set("sig_x", self.background_material.sig_x)
            cell.set("sig_y", self.background_material.sig_y)

        #iterate materials
        for material in self.materials: 

            indices = []
            for cell in self.QT.get_leafs_inside_polygon(material.polygon):

                i = int(cell.get("idx"))
                indices.append(i)
                background_indices.remove(i)

                #set material parameters
                cell.set("eps_r_x", material.eps_r_x)
                cell.set("eps_r_y", material.eps_r_y)
                cell.set("sig_x", material.sig_x)
                cell.set("sig_y", material.sig_y)

            #assign cell indices to the material
            material.set_indices(indices)

            self.logger(f"progress - '{material.label}' assigned to {len(indices)} cells")

        #set indices for background material
        self.background_material.set_indices(background_indices)
        self.logger(f"progress - '{self.background_material.label}' assigned to {len(background_indices)} cells")





    def _setup_interfaces(self):

        if not len(self.interfaces):
            self.logger("SKIPPING INTERFACE SETUP")
            return

        self.logger("INTERFACE SETUP")

        #get the cells
        leafs = self.QT.get_leafs()

        #iterate interfaces
        for inter in self.interfaces:

            #reset interface
            inter.reset()

            #iterate cell indices of material
            for i in inter.material_a.indices:

                #iterate neighbours of cell in 'material_a'
                for neighbor in leafs[i].get_neighbors():
                    j = int(neighbor.get("idx"))

                    #if neighbour in 'material_b', they form an interface
                    if j in inter.material_b.indices:
                        inter.add((i, j))
                        inter.add((j, i))

            self.logger(f"progress - Interface '{inter.label}' with {len(inter.pairs)} cells")





    def _setup_ports(self):

        if not len(self.ports):
            self.logger("SKIPPING PORT SETUP")
            return 

        self.logger("PORT SETUP")

        #iterate ports
        for prt in self.ports:

            #place source in cell
            if prt.p_src is not None:
                idx = int(self.QT.get_closest_leaf(prt.p_src).get("idx"))
                prt.set_src_cell_idx(idx)
                self.logger(f"progress - Port '{prt.label}' (src) -> cell {idx}")

            #place sink in cell
            if prt.p_snk is not None:
                idx = int(self.QT.get_closest_leaf(prt.p_snk).get("idx"))
                prt.set_snk_cell_idx(idx)
                self.logger(f"progress - Port '{prt.label}' (snk) -> cell {idx}")
















    # @timer
    def setup(self):
        """
        frequency independent simulation setup
        """
        self._setup_logging()
        self._setup_mesh()
        self._setup_materials()
        self._setup_interfaces()
        self._setup_ports()








    def reset(self):
        """
        reset and recomput everything except the mesh and logging
        """
        self._setup_materials()
        self._setup_interfaces()
        self._setup_ports()







    def _build_system_boundary(self, omega):

        #setup linear system
        n_cells = len(self.QT)
        Ab = lil_matrix((n_cells, n_cells), dtype=np.complex128)
        bb = np.zeros(n_cells, dtype=np.complex128)

        #helper functions for material parameters
        def g_x(cell, omega):
            return cell.get("sig_x") + 1j * omega * self.eps_0 * cell.get("eps_r_x")

        def g_y(cell, omega):
            return cell.get("sig_y") + 1j * omega * self.eps_0 * cell.get("eps_r_y")

        #iterate the boundary cells
        for cell in self.QT.get_leafs_at_boundary():

            i = int(cell.get("idx"))
            w_i, h_i = cell.size

            #top boundary cells (N), default is neumann
            if cell.is_boundary_N():

                kind, value = self.boundary_condition_y_max

                if kind == "dirichlet":
                    a_ii = 2 * w_i / h_i * g_y(cell, omega)   
                    Ab[i, i] -= a_ii
                    bb[i] -= value * a_ii

                elif kind == "periodic":
                    vertical_neighbors, _ = cell.get_periodic_neighbors()

                    #iterate the periodic neighbors (assuming same size)
                    for neighbor in vertical_neighbors:

                        #get index of neighbor cell (j)
                        j = int(neighbor.get("idx"))

                        #get neighbor cell (j) parameters
                        w_j, h_j = neighbor.size

                        #compute effective conductance
                        g_y_ij = 2/(1/g_y(cell, omega) + 1/g_y(neighbor, omega))

                        #construct matrix entries
                        a_ij = w_i / h_i * g_y_ij
                        Ab[i, i] -= a_ij
                        Ab[i, j] += a_ij

            #bottom boundary cells (S), default is neumann
            elif cell.is_boundary_S():

                kind, value = self.boundary_condition_y_min

                if kind == "dirichlet":
                    a_ii = 2 * w_i / h_i * g_y(cell, omega)   
                    Ab[i, i] -= a_ii
                    bb[i] -= value * a_ii

                elif kind == "periodic":
                    vertical_neighbors, _ = cell.get_periodic_neighbors()

                    #iterate the periodic neighbors (assuming same size)
                    for neighbor in vertical_neighbors:

                        #get index of neighbor cell (j)
                        j = int(neighbor.get("idx"))

                        #get neighbor cell (j) parameters
                        w_j, h_j = neighbor.size

                        #compute effective conductance
                        g_y_ij = 2/(1/g_y(cell, omega) + 1/g_y(neighbor, omega))

                        #construct matrix entries
                        a_ij = w_i / h_i * g_y_ij
                        Ab[i, i] -= a_ij
                        Ab[i, j] += a_ij

            #left boundary cells (W), default is neumann
            if cell.is_boundary_W():

                kind, value = self.boundary_condition_x_min

                if kind == "dirichlet":
                    a_ii = 2 * h_i / w_i * g_x(cell, omega)  
                    Ab[i, i] -= a_ii
                    bb[i] -= value * a_ii

                elif kind == "periodic":

                    _, horizontal_neighbors = cell.get_periodic_neighbors()

                    #iterate the periodic neighbors (assuming same size)
                    for neighbor in horizontal_neighbors:

                        #get index of neighbor cell (j)
                        j = int(neighbor.get("idx"))

                        #get neighbor cell (j) parameters
                        w_j, h_j = neighbor.size

                        #compute effective conductance
                        g_x_ij = 2/(1/g_x(cell, omega) + 1/g_x(neighbor, omega))

                        #construct matrix entries
                        a_ij = h_i / w_i * g_x_ij
                        Ab[i, i] -= a_ij
                        Ab[i, j] += a_ij

            #right boundary cells (E), default is neumann
            elif cell.is_boundary_E():

                kind, value = self.boundary_condition_x_max
                
                if kind == "dirichlet":
                    a_ii = 2 * h_i / w_i * g_x(cell, omega)   
                    Ab[i, i] -= a_ii
                    bb[i] -= value * a_ii

                elif kind == "periodic":

                    _, horizontal_neighbors = cell.get_periodic_neighbors()

                    #iterate the periodic neighbors (assuming same size)
                    for neighbor in horizontal_neighbors:

                        #get index of neighbor cell (j)
                        j = int(neighbor.get("idx"))

                        #get neighbor cell (j) parameters
                        w_j, h_j = neighbor.size

                        #compute effective conductance
                        g_x_ij = 2/(1/g_x(cell, omega) + 1/g_x(neighbor, omega))

                        #construct matrix entries
                        a_ij = h_i / w_i * g_x_ij
                        Ab[i, i] -= a_ij
                        Ab[i, j] += a_ij

        return Ab.tocsr(), bb





























    def _build_system_homogeneous(self, omega):
        """
        build homogeneous linear system without boundary conditions and ports
        """

        #setup linear system
        n_cells = len(self.QT)
        Ah = lil_matrix((n_cells, n_cells), dtype=np.complex128)

        #helper functions for material parameters
        def g_x(cell, omega):
            return cell.get("sig_x") + 1j * omega * self.eps_0 * cell.get("eps_r_x")

        def g_y(cell, omega):
            return cell.get("sig_y") + 1j * omega * self.eps_0 * cell.get("eps_r_y")
       
        #iterate cells to incorporate boundary conditions and materials
        for i, cell in performance_enumerate(self.QT.get_leafs(), log=self.log):
            
            #get cell (i) parameters
            w_i, h_i = cell.size
            
            #vertical neighbors 
            for sgn, neighbors in zip([1.0, -1.0], [cell.get_neighbors_N(), cell.get_neighbors_S()]):
                for neighbor in neighbors:

                    #get index of neighbor cell (j)
                    j = int(neighbor.get("idx"))

                    #get neighbor cell (j) parameters
                    w_j, h_j = neighbor.size

                    #compute effective conductance
                    g_y_ij = (h_i + h_j) / (h_i/g_y(cell, omega) + h_j/g_y(neighbor, omega))

                    #check if interface between (i) and (j) is defined
                    for inter in self.interfaces:
                        if inter.is_interface((i, j)):

                            #interface parameters (thickness and complex conductivity)
                            d, g_f = inter.delta, 1j * omega * inter.cap

                            #inject complex interface conductivity
                            g_y_ij = (h_i + h_j) / ((h_i-d)/g_y(cell, omega) + 1/g_f + (h_j-d)/g_y(neighbor, omega))  
                            
                            break

                    #relevant cells (including shared neighbors)
                    relevant_cells = [cell, neighbor, *cell.get_shared_neighbors(neighbor)]

                    #no neighbors neighbors (same size)
                    if len(relevant_cells) == 2:
    
                        #construct matrix entries 
                        a_ij = w_i / h_i * g_y_ij
                        Ah[i, i] -= a_ij
                        Ah[i, j] += a_ij
                    
                    else:
                        #relevant cell indices
                        indices = [int(c.get("idx")) for c in relevant_cells]
                        
                        #assemble least squares system
                        M = np.array([[1, *c.center] for c in relevant_cells])

                        #compute least squares interpolation matrix
                        A_ij = sgn * g_y_ij * min(w_i, w_j) * np.linalg.solve(np.dot(M.T, M), M.T)[2, :]

                        #incorporate into system
                        for j, a_ij in zip(indices, A_ij):
                            Ah[i, j] += a_ij 


            #horizontal neighbors 
            for sgn, neighbors in zip([1.0, -1.0], [cell.get_neighbors_E(), cell.get_neighbors_W()]):
                for neighbor in neighbors:

                    #get index of neighbor cell (j)
                    j = int(neighbor.get("idx"))

                    #get neighbor cell (j) parameters
                    w_j, h_j = neighbor.size

                    #compute effective conductance
                    g_x_ij = (w_i + w_j) / (w_i/g_x(cell, omega) + w_j/g_x(neighbor, omega))

                    #check if interface between (i) and (j) is defined
                    for inter in self.interfaces:
                        if inter.is_interface((i, j)):

                            #interface parameters (thickness and complex conductivity)
                            d, g_f = inter.delta, 1j * omega * inter.cap

                            #inject complex interface conductivity
                            g_x_ij = (w_i + w_j) / ((w_i-d)/g_x(cell, omega) + 1/g_f + (w_j-d)/g_x(neighbor, omega))
                            
                            break

                    #relevant cells (including shared neighbors)
                    relevant_cells = [cell, neighbor, *cell.get_shared_neighbors(neighbor)]

                    #no neighbors neighbors (same size)
                    if len(relevant_cells) == 2:
    
                        #construct matrix entries
                        a_ij = h_i / w_i * g_x_ij
                        Ah[i, i] -= a_ij 
                        Ah[i, j] += a_ij
                    
                    else:
                        #relevant cell indices
                        indices = [int(c.get("idx")) for c in relevant_cells]
                        
                        #assemble least squares system
                        M = np.array([[1, *c.center] for c in relevant_cells])

                        #compute least squares interpolation matrix
                        A_ij = sgn * g_x_ij * min(h_i, h_j) * np.linalg.solve(np.dot(M.T, M), M.T)[1, :]

                        #incorporate into system
                        for j, a_ij in zip(indices, A_ij):
                            Ah[i, j] += a_ij

        return Ah.tocsr()














    def _build_system_inhomogeneous(self):

        #setup linear system
        n_cells = len(self.QT)
        As = lil_matrix((n_cells, n_cells), dtype=np.complex128)
        bs = np.zeros(n_cells, dtype=np.complex128)

        #iterate ports to incorporate sources and constraints
        for prt in self.ports:
            
            #cell indices of port (src) and (snk) points
            i = prt.src_cell_idx
            j = prt.snk_cell_idx

            #current injection into cell if specified by port
            if prt.I_src is not None and prt.active :
                if i is not None:
                    bs[i] -= prt.I_src
                if j is not None:
                    bs[j] += prt.I_src

            #enforce potential difference (constraint) if specified by port
            if prt.V_src is not None:
                if i is not None:
                    As[i, i] += 1.0
                    bs[i] += prt.V_src
                if j is not None:
                    As[j, j] += 1.0
                    bs[j] -= prt.V_src   
                if i is not None and j is not None:
                    As[i, j] -= 1.0
                    As[j, i] -= 1.0

        return As.tocsr(), bs














    # @timer
    def solve(self, omega):
        """
        solve the system for a given frequency
        """

        self.logger(f"SOLVING SYSTEM for omega = {omega}Hz")
        
        #build boundary condition part
        Ab, bb = self._build_system_boundary(omega)

        #build homogeneous part of frequency dependent complex linear system
        Ah = self._build_system_homogeneous(omega)

        #build inhomogeneous system part
        As, bs = self._build_system_inhomogeneous()

        #solve complex linear system
        Phi = spsolve(Ah+Ab+As, bb+bs)

        self.logger(f"progress - System solved")

        #iterate cells and assign potential from solution
        for i, cell in enumerate(self.QT.get_leafs()): 
            cell.set("Phi_re", Phi[i].real)
            cell.set("Phi_im", Phi[i].imag)

        self.logger(f"progress - Cells updated")

        #set port voltages from solution
        for prt in self.ports:

            #get cell indices of port points
            i = prt.src_cell_idx
            j = prt.snk_cell_idx

            #get potentials at port cells
            V_i = 0.0 if i is None else Phi[i]
            V_j = 0.0 if j is None else Phi[j]

            #set port voltage as potential difference
            prt.V = V_i - V_j

            self.logger(f"progress - Phi: port '{prt.label}' <- cells {i, j}")








    def solve_multiport(self, omega):
        """
        solve the system for a given frequency for all ports
        """

        self.logger(f"SOLVING SYSTEM for omega = {omega}Hz")
        
        #build boundary condition part
        Ab, bb = self._build_system_boundary(omega)

        #build homogeneous part of frequency dependent complex linear system
        Ah = self._build_system_homogeneous(omega)

        #get relevant current ports
        relevant_ports = [prt for prt in self.ports if prt.I_src is not None]

        #total number of ports
        n_prt = len(relevant_ports)
        n_cells = len(self.QT)
        B1 = np.zeros((n_cells, n_prt), dtype=np.complex128)
        B2 = np.zeros((n_cells, n_prt), dtype=np.complex128)

        #initialize impedance matrix
        Z = np.zeros((n_prt, n_prt), dtype=np.complex128)

        self.logger(f"progress - solving for {len(relevant_ports)} relevant ports")

        #iterate ports and set excitations
        for k, prt in enumerate(relevant_ports):

            #add inhomogenity from boundary conditions
            B1[:, k] += bb

            #cell indices of port (src) and (snk) points
            i = prt.src_cell_idx
            j = prt.snk_cell_idx

            #add inhomogenity from port
            if i is not None:
                B1[i, k] -= prt.I_src
                B2[i, k] -= prt.I_src
            if j is not None:
                B1[j, k] += prt.I_src
                B2[j, k] += prt.I_src

        #solve and map to impedance matrix
        Z = - np.dot(B2.T, spsolve(Ah+Ab, B1))

        #return the impedance matrix for given frequency
        return Z







    def compute_E_J(self, omega):

        self.logger("COMPUTING GRADIENTS")

        #iterate cells and compute gradients
        for _, cell in performance_enumerate(self.QT.get_leafs(), log=self.log):

            #compute gradient of potential with interpolation order 2
            grad_x_re, grad_y_re = cell.grad("Phi_re", 2)
            grad_x_im, grad_y_im = cell.grad("Phi_im", 2)

            #set electric field
            cell.set("E_x_re", -grad_x_re)
            cell.set("E_x_im", -grad_x_im)
            cell.set("E_y_re", -grad_y_re)
            cell.set("E_y_im", -grad_y_im)

            #complex valued gradient
            grad_x = grad_x_re + 1j * grad_x_im
            grad_y = grad_y_re + 1j * grad_y_im
            
            #complex conductivity of cell
            g_x = cell.get("sig_x") + 1j * omega * self.eps_0 * cell.get("eps_r_x")
            g_y = cell.get("sig_y") + 1j * omega * self.eps_0 * cell.get("eps_r_y")

            #set current density
            cell.set("J_x_re", -(g_x*grad_x).real)
            cell.set("J_x_im", -(g_x*grad_x).imag)
            cell.set("J_y_re", -(g_y*grad_y).real)
            cell.set("J_y_im", -(g_y*grad_y).imag)

        self.logger("progress - gradients computed")



    def map_to_grid(self, grid, parameter="Phi_re"):

        #unpack grid
        grid_x, grid_y = grid
        nx, ny = grid_x.shape

        values_flat = np.zeros(nx*ny)

        for i, point in enumerate(zip(grid_x.flatten(), grid_y.flatten())):
            cell = self.QT.get_closest_leaf(point)
            values_flat[i] = cell.get(parameter)

        return values_flat.reshape((nx, ny))



















