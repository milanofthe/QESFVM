###################################################################################
##
##                    'QuadTree' and 'Cell' definition
##
##                          Milan Rother 2023/24
##
###################################################################################


# IMPORTS =========================================================================

import numpy as np




# HELPER FUNCTIONS ================================================================

def project_point_to_line(point, line_start, line_end):
    """
    Projects a point orthogonally onto a line segment. 
    If the projected point is not on the line segment, 
    clip it to the line.
    
    INPUTS : 
        point      : (tuple of floats) point to project
        line_start : (tuple of floats) starting point of line segment
        line_end   : (tuple of floats) end point of line segment 
    """

    # Ensure correct dimensions
    if not all(len(v) == 2 for v in [point, line_start, line_end]):
        return None 

    # Calculate direction vector of the line
    line_direction = np.array(line_end) - np.array(line_start) 

    # Pre-calculate squared length of the line segment
    line_squared_length = np.dot(line_direction, line_direction)

    # Vector from line's start to the point to project
    point_to_line = np.array(point) - np.array(line_start)

    # Projection distance along the line
    dot_product = np.dot(point_to_line, line_direction)
    projection_distance = dot_product / line_squared_length

    # Clamp for robustness
    projection_distance = np.clip(projection_distance, 0.0, 1.0)

    # Calculate the projected point
    projected_point = np.array(line_start) + projection_distance * line_direction 

    return projected_point 


def is_point_inside_polygon(point, polygon):
    """
    Checks if a point is inside a polygon using the winding number algorithm.

    INPUTS : 
        point   : (tuple of floats) point to check
        polygon : (list of tuples of floats) non closed path that defines the polygon
    """
    
    # Quick bounding box check
    x_coords, y_coords = zip(*polygon) 
    if (point[0] < min(x_coords) or 
        point[0] > max(x_coords) or 
        point[1] < min(y_coords) or 
        point[1] > max(y_coords)):
        return False

    # Winding number calculation
    winding_number = 0

    for v1, v2 in zip(polygon, polygon[-1:]+polygon[:-1]):

        if v1[1] <= point[1]:
            if v2[1] > point[1]:  
                # An upward crossing
                if (v2[0] - v1[0]) * (point[1] - v1[1]) - (point[0] - v1[0]) * (v2[1] - v1[1]) > 0:
                    winding_number += 1
        else:  
            if v2[1] <= point[1]:  
                # A downward crossing
                if (v2[0] - v1[0]) * (point[1] - v1[1]) - (point[0] - v1[0]) * (v2[1] - v1[1]) < 0:
                    winding_number -= 1

    return winding_number != 0














# Cell CLASS ======================================================================

class Cell:
    """
    This is the basic building block for a 2D N-Tree datastructure.

    INPUTS : 
        center : (tuple of floats) center coordinate of the cell
        size   : (tuple of floats) width and height of the cell
        level  : (int) initialization level / refinement level of the cell
        parent : (Cell) reference to parent cell in tree
    """
    
    def __init__(self, center=(0,0), size=(1,1), level=0, parent=None):
        
        #cell geometry
        self.center = np.asarray(center)
        self.size = np.asarray(size)
        
        #ancestor
        self.parent = parent
        
        #keep track of refinement level
        self.level = level

        #lists of neighboring cells
        self.neighbors_N = []
        self.neighbors_S = []
        self.neighbors_W = []
        self.neighbors_E = []
        
        #child cells
        self.children = []
        
        #parameters or data for cell
        self.parameters = {}


    def split(self, nx=2, ny=2):
        """
        Split cell by creating child cells and dynamically updating 
        the neighbors and the neighbors neighbors.
        
        INPUTS : 
            nx : (int) number of splits in x-axis
            ny : (int) number of splits in y-axis
        """
        
        #clear existing children
        self.merge()
        
        #get cell width, height and location
        w, h = self.size
        x, y = self.center

        #compute size of new children
        _size = (w/nx, h/ny)

        #create new children
        for i in range(nx):
            _x = x + w*((i + 0.5)/nx - 0.5)
            for j in range(ny):
                _y = y + h*((j + 0.5)/ny - 0.5)

                #assemble center of child
                _center = (_x, _y)
                
                #add child cell to children, increment refinement level and add parent
                self.children.append(Cell(center=_center, 
                                          size=_size, 
                                          parent=self, 
                                          level=self.level+1))
        
        #update neighbors of new children
        for i, child in enumerate(self.children):
            
            #collect relevant siblings, children NSWE of child
            relevant_siblings = []
            if i-1 >= 0:
                relevant_siblings.append(self.children[i-1]) 
            if i+1 < nx*ny:
                relevant_siblings.append(self.children[i+1]) 
            if i-ny >= 0:
                relevant_siblings.append(self.children[i-ny]) 
            if i+ny < nx*ny:
                relevant_siblings.append(self.children[i+ny]) 
            
            #assemble possible neighbor candidates and check them
            neighbor_candidates = self.get_neighbors() + relevant_siblings
            child.find_neighbors(neighbor_candidates)
            
        #add new children as neighbors of old neighbors
        for neighbor in self.get_neighbors():
            
            #assemble possible neighbor candidates and check them
            neighbor_candidates = neighbor.get_neighbors() + self.children
            neighbor.find_neighbors(neighbor_candidates)


    def find_neighbors(self, cells, tol=1e-12):
        """
        search for neighbor cells that are leaf cells and 
        update the neighbors of the cell accordingly
    
        INPUTS : 
            cells : (list of cells) cells to check
            tol   : (float) numerical tolerance for neighbor checking
        """

        #get cell boundary (nsew)
        x, y = self.center
        e, n = self.center + self.size/2
        w, s = self.center - self.size/2

        #clear existing neighbors    
        self.neighbors_N.clear()
        self.neighbors_S.clear()
        self.neighbors_W.clear()
        self.neighbors_E.clear()
        
        #iterate cells to check for "neighborhood"
        for i, cell in enumerate(cells):

            #skip if cell has children (is not leaf) or is duplicate
            if cell.children or cell in cells[(i+1):]:
                continue

            #get other cell boundaries (nsew)
            _x, _y = cell.center
            _e, _n = cell.center + cell.size/2
            _w, _s = cell.center - cell.size/2

            #check if cells touch and in which direction 
            if (_w+tol < x < _e-tol) or (w+tol < _x < e-tol):
                if abs(n - _s) < tol:
                    self.neighbors_N.append(cell)
                elif abs(s - _n) < tol:
                    self.neighbors_S.append(cell)
            elif (_s+tol < y < _n-tol) or (s+tol < _y < n-tol):
                if abs(e - _w) < tol:
                    self.neighbors_E.append(cell)
                elif abs(w - _e) < tol:
                    self.neighbors_W.append(cell)


    def merge(self, preserve_parameters=[]):
        """
        Recursively merge all children and make cell a leaf while perserving 
        the children parameters through computing the mean value

        INPUTS : 
            preserve_parameters : (list of strings) list of cell parmeters that should be mapped to parent
        """

        # catch case for no children
        if not self.children:
            return 

        # recursively merge children's children
        for child in self.children:
            child.merge()  

        # preserve parameters to parent -> compute mean from children parameters
        for param in preserve_parameters:
            self.set(param, np.mean([child.get(param)  for child in self.children]))

        # get all the children neighbors
        neighbor_candidates = [cell for cell in child.get_neighbors() 
                               if cell not in self.children 
                               for child in self.children]

        # update neighbors from child neighbors
        self.find_neighbors(neighbor_candidates)

        # update the neighbors
        for neighbor in self.get_neighbors():

            # get neighbor candidates of neighbors including self but not original children
            neighbor_candidates = [self] + [cell for cell in neighbor.get_neighbors() 
                                            if cell not in self.children]

            # update neighbors neighbors
            neighbor.find_neighbors(neighbor_candidates)

        # delete all children
        self.children.clear()


    def contains_point(self, point, tol=1e-12):
        """
        check if a point is inside of the cell

        INPUTS : 
            point : (tuple of floats) x-y coordinate point to check
        """
        return ((point[0] >= self.center[0] - self.size[0]/2 - tol) and 
                (point[0] <= self.center[0] + self.size[0]/2 + tol) and
                (point[1] >= self.center[1] - self.size[1]/2 - tol) and
                (point[1] <= self.center[1] + self.size[1]/2 + tol))


    def is_cut_by_line(self, line_start, line_end, tol=1e-12):
        """
        check if the cell is cut by a line segment defined by the start and end points

        INPUTS : 
            line_start : (tuple of floats) starting point of the line segment
            line_end   : (tuple of floats) end point of the line segment
            tol        : (float) tolerance for checking if cell is cut
        """

        # unpack cell geometry
        x, y = self.center
        w, h = self.size

        # bounding box check of line segment
        if (min(line_start[0], line_end[0]) > x + w/2 + tol or
            max(line_start[0], line_end[0]) < x - w/2 - tol or 
            min(line_start[1], line_end[1]) > y + h/2 + tol or 
            max(line_start[1], line_end[1]) < y - h/2 - tol):
            return False

        # check if corner projections are within cell
        return (self.contains_point(project_point_to_line([x - w/2, y + h/2], line_start, line_end), tol) or 
                self.contains_point(project_point_to_line([x + w/2, y + h/2], line_start, line_end), tol) or
                self.contains_point(project_point_to_line([x - w/2, y - h/2], line_start, line_end), tol) or 
                self.contains_point(project_point_to_line([x + w/2, y - h/2], line_start, line_end), tol))


    def is_inside_polygon(self, polygon):
        return is_point_inside_polygon(self.center, polygon)


    def get_periodic_neighbors(self):

        #get the root cell
        root = self.get_root()

        #bounding box from root size
        bb_w, bb_h = root.size 

        #get cell geometry
        x, y = self.center
        w, h = self.size

        #collect candidate points for periodic neighbor checks
        points_vertical = []
        points_horizontal = []

        if self.is_boundary_E():
            points_horizontal.extend([(x + 3/4 * w - bb_w, y + 1/4 * h), 
                                      (x + 3/4 * w - bb_w, y - 1/4 * h)])

        if self.is_boundary_W():
            points_horizontal.extend([(x - 3/4 * w + bb_w, y + 1/4 * h), 
                                      (x - 3/4 * w + bb_w, y - 1/4 * h)])

        if self.is_boundary_N():
            points_vertical.extend([(x + 1/4 * w, y + 3/4 * h - bb_h), 
                                    (x - 1/4 * w, y + 3/4 * h - bb_h)])

        if self.is_boundary_S():
            points_vertical.extend([(x + 1/4 * w, y - 3/4 * h + bb_h), 
                                    (x - 1/4 * w, y - 3/4 * h + bb_h)])

        #get the leafs which contain the candidate points
        periodic_neighbors_vertical = []
        periodic_neighbors_horizontal = []

        for p in points_vertical:
            candidate = root.get_closest_leaf(p)
            if candidate not in periodic_neighbors_vertical:
                periodic_neighbors_vertical.append(candidate)

        for p in points_horizontal:
            candidate = root.get_closest_leaf(p)
            if candidate not in periodic_neighbors_horizontal:
                periodic_neighbors_horizontal.append(candidate)

        return periodic_neighbors_vertical, periodic_neighbors_horizontal


    def get_root(self):
        """
        recursively traverse the tree upward to retrieve the root cell
        """
        if self.parent is None:
            return self
        else:
            return self.parent.get_root()


    def get_leafs(self):        
        """
        return all cells from lowest refinement level (tree leafs)
        """
        cells = [self] if not self.children else []
        for child in self.children:
            cells += child.get_leafs()
        return cells


    def get_neighbors(self):
        """
        returns list of all neighbor cells
        """
        return (self.neighbors_N + 
                self.neighbors_S + 
                self.neighbors_W + 
                self.neighbors_E)


    def get_neighbors_N(self):
        return self.neighbors_N


    def get_neighbors_S(self):
        return self.neighbors_S


    def get_neighbors_W(self):
        return self.neighbors_W


    def get_neighbors_E(self):
        return self.neighbors_E


    def get_shared_neighbors(self, other):
        """
        Returns list of all cells that are both neighbors 
        of self and of other, e.g. shared neighbors
        
        INPUTS : 
            other : (Cell) the other cell    
        """
        self_neighbors = self.get_neighbors()
        other_neighbors = other.get_neighbors()
        return [cell for cell in self_neighbors if cell in other_neighbors]


    def get_neighbors_of_degree(self, degree=2):
        """
        Retrieve cells from the direct neighborhood up to a certain 
        degree of connectedness / degree of neighborhood. 
        Excluding the cell itself.
    
        INPUTS : 
            degree : (int) degree of connectedness up to which cells are retrieved
        """
        k = 0
        candidates = [self]
        for _ in range(degree):
            _candidates = candidates[k:]
            k = len(candidates)
            for cell in _candidates:
                for neighbor in cell.get_neighbors():
                    if neighbor not in candidates:
                        candidates.append(neighbor)
        return candidates[1:]


    def get_combined_neighbors_of_degree(self, other, degree=2):
        """
        Retrieve cells from the direct neighborhood of two cells up 
        to a certain degree of connectedness / degree of neighborhood. 
        Excluding the cells themself.
    
        INPUTS : 
            other  : (Cell) the other cell
            degree : (int) degree of connectedness up to which cells are retrieved
        """
        k = 0
        candidates = [self, other]
        for _ in range(degree):
            _candidates = candidates[k:]
            k = len(candidates)
            for cell in _candidates:
                for neighbor in cell.get_neighbors():
                    if neighbor not in candidates:
                        candidates.append(neighbor)
        return candidates[2:]


    def get_neighborhood(self, n=12):
        """
        Retrieve some number of cells from the direct neighborhood 
        with in creasing degree of connectedness. Excluding the cell itself.
    
        INPUTS : 
            n : (int) number of cells to retrieve
        """
        k = 0
        candidates = [self]
        while True:
            _candidates = candidates[k:]
            k = len(candidates)
            for cell in _candidates:
                for neighbor in cell.get_neighbors():
                    if neighbor not in candidates:
                        candidates.append(neighbor)
                        if len(candidates) >= n + 1:
                            return candidates[1:]


    def get_combined_neighborhood(self, other, n=12):
        """
        Retrieve some number of cells from the direct neighborhood of two cells
        with in creasing degree of connectedness. Excluding the two cells themself.
    
        INPUTS : 
            other : (Cell) the other cell
            n     : (int) number of cells to retrieve
        """
        k = 0
        candidates = [self, other]
        while True:
            _candidates = candidates[k:]
            k = len(candidates)
            for cell in _candidates:
                for neighbor in cell.get_neighbors():
                    if neighbor not in candidates:
                        candidates.append(neighbor)
                        if len(candidates) >= n + 2:
                            return candidates[2:]


    def get_closest_leaf(self, point):
        """
        get the leaf cell that contains the point

        INPUTS : 
            point : (tuple of floats)
        """

        #terminate recursion if leaf is reached
        if not self.children:
            return self

        #iterate children recursively and check if they contain the point
        for child in self.children:
            if child.contains_point(point):
                return child.get_closest_leaf(point)

        # point is outside the cell, return the cell itself
        return self


    def is_boundary(self):
        """
        check if the cell is a boundary cell 
        (it has no neighbors in at least one direction)
        """
        return not (self.neighbors_N and self.neighbors_S and self.neighbors_W and self.neighbors_E)


    def is_boundary_N(self):
        return not self.neighbors_N


    def is_boundary_S(self):
        return not self.neighbors_S


    def is_boundary_W(self):
        return not self.neighbors_W


    def is_boundary_E(self):
        return not self.neighbors_E


    def is_balanced(self):
        """
        check if cell has more then 2 neighbors in one direction (NSWE)
        if yes, it is not balanced
        """
        return (len(self.neighbors_N) <= 2 and 
                len(self.neighbors_S) <= 2 and 
                len(self.neighbors_W) <= 2 and 
                len(self.neighbors_E) <= 2)


    def set(self, key, value):
        """
        set a value in the cell parameter dict
        """
        self.parameters[key] = value


    def get(self, key):
        """
        retrieve a value from the cell parameter dict
        """
        return self.parameters.get(key, 0.0)


    def grad(self, parameter=None, order=2):
        """
        compute gradient of a parameter for the cell at the cell center 
        using a bivariate polynomial fit to neighboring cells with 
        distance weighted least squares
        """

        #retrieve neighborhood cells for interpolation
        relevant_cells = [self, *self.get_neighborhood((order+1)**2)]

        #precompute the relative centers for evaluation
        relative_points = [c.center-self.center for c in relevant_cells]

        #build linear system depending on available cells (biquadratic fit) 
        A = [[(x**i * y**j) for j in range(order+1) for i in range(order+1)] for x, y in relative_points]
        b = [c.get(parameter) for c in relevant_cells]
            
        #solve least squares problem for bilinear approximation of gradient
        coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)    

        #unpack gradient at cell center from result
        grd_x = coeffs[1]
        grd_y = coeffs[order+1]

        return grd_x, grd_y






# QuadTree CLASS ==================================================================

class QuadTree:
    """
    class for managing the 2D Quadtree datastructure

    INPUTS : 
        bounding_box : (list of tuples) bounding box of the quadtree root cell
        n_initial_x  : (int) number of initial root cell splits along x-axis
        n_initial_y  : (int) number of initial root cell splits along y-axis
    """

    def __init__(self, bounding_box=[(-1, -1), (1, 1)], n_initial_x=2, n_initial_y=2):

        self.bounding_box = bounding_box

        #determine root cell size and center from  bounding box
        bb_x, bb_y = zip(*self.bounding_box)
        center = sum(bb_x)/2, sum(bb_y)/2
        size = abs(min(bb_x) - max(bb_x)), abs(min(bb_y) - max(bb_y))

        #initialize quadtree root cell
        self.root = Cell(center, size)

        #initial root cell splits
        self.root.split(n_initial_x, n_initial_y)


    def __len__(self):
        return len(self.get_leafs())


    def get_leafs(self):
        return self.root.get_leafs()


    def get_closest_leaf(self, point):
        return self.root.get_closest_leaf(point)


    def balance(self):
        """
        Ballance all leaf cells of the quadtree by splitting the 
        cells that have more then 2 neighbors in some direction 
        (sometimes this is also called a graded quadtree).
        """

        #balancing flag
        needs_balancing = True

        #balance individual cells until all leafs are balanced
        while needs_balancing:

            needs_balancing = False

            #get the leafs
            leafs = self.get_leafs()

            #iterate all relevant leaf cells
            for cell in leafs:

                if not cell.is_balanced():
                    cell.split(2, 2)
                    needs_balancing = True


    def refine_boundary(self,
                        x_min=False,
                        x_max=False,
                        y_min=False,
                        y_max=False,
                        min_size=0.0):
        """
        Automatically refine leaf cells based on the selected mode

        INPUTS : 
            x_min       : (bool) quadtree refinement at left boundary
            x_max       : (bool) quadtree refinement at right boundary
            y_min       : (bool) quadtree refinement at bottom boundary
            y_max       : (bool) quadtree refinement at top boundary
            min_size    : (float) sets smallest allowed cell size
        """

        for cell in self.get_leafs_at_boundary():

            if ((x_min and cell.is_boundary_W()) or 
                (x_max and cell.is_boundary_E()) or 
                (y_min and cell.is_boundary_S()) or 
                (y_max and cell.is_boundary_N())):
                cell.split(2, 2)


    def refine_edge(self,
                    segments,
                    min_size=0.0,
                    tol=1e-12):
        """
        Automatically refine leaf cells based on the selected mode and geometry. 
        The geometry is provided in the format of line segments that consist of 
        two points (x-y-coords) each. The method checks for all leaf cells if they 
        are intersected by the segments.

        INPUTS : 
            segments : (list of list tuples) set of line segments made of two points each that form path
            min_size : (float) sets smallest allowed cell size
            tol      : (float) numerical tolerance for checking if a cell is cut by the segment
        """ 

        for cell in self.get_leafs_cut_by_segments(segments, tol):
            if min(cell.size) > min_size:
                cell.split(2, 2)


    def get_leafs_cut_by_segments(self, segments, tol=1e-12):
        """
        Retrieve all the leaf cells that are cut by the line segments with some tolerance.

        INPUTS : 
            segments : (list of lists of tuples of floats) list of line segments that are defined by two points each
            tol      : (float) numerical tolerance for checking if a cell is cut by the segment
        """

        relevant_leafs = []
        for cell in self.get_leafs():
            for p1, p2 in segments:
                if cell.is_cut_by_line(p1, p2, tol):
                    relevant_leafs.append(cell)
                    break
        return relevant_leafs


    def get_leafs_at_boundary(self):
        """
        Retrieve the leaf cells at the quadtree boundary.
        """
        return [cell for cell in self.get_leafs() if cell.is_boundary()]


    def get_leafs_inside_polygon(self, polygon):
        """
        Retrieve the leaf cells that are within a polygon

        INPUTS : 
            polygon : (list of tuples of floats) non closed path that defines the polygon
        """
        return [cell for cell in self.get_leafs() if cell.is_inside_polygon(polygon)]


    def get_leafs_inside_polygons(self, polygons=[]):
        """
        Retrieve the leaf cells that are within a cutset of multiple polygons

        INPUTS : 
            polygons : (list of lists of tuples of floats) multiple non closed paths that define polygons
        """
        return [cell for cell in self.get_leafs() if np.all([cell.is_inside_polygon(poly) for poly in polygons])]


    def get_leafs_outside_polygon(self, polygon):
        """
        Retrieve the leaf cells that are outside of a polygon

        INPUTS : 
            polygon : (list of tuples of floats) non closed path that defines the polygon
        """
        return [cell for cell in self.get_leafs() if not cell.is_inside_polygon(polygon)]


    def get_leafs_outside_polygons(self, polygons=[]):
        """
        Retrieve the leaf cells that are outside a cutset of multiple polygons

        INPUTS : 
            polygons : (list of lists of tuples of floats) multiple non closed paths that define polygons
        """
        return [cell for cell in self.get_leafs() if not np.any([cell.is_inside_polygon(poly) for poly in polygons])]


    def get_leafs_at_polygon_boundary(self, polygon):
        """
        Retrieve the leaf cells that are directly at the boudnary of a polygon and either 
        inside or outside the polygon, both lists are returned as a tuple

        INPUTS : 
            polygon : (list of tuples of floats) non closed path that defines the polygon            
        """

        cells_at_boundary_inside = []
        cells_at_boundary_outside = []

        for cell in self.get_leafs():
            if cell.is_inside_polygon(polygon):
                for neighbor in cell.get_neighbors():
                    if not neighbor.is_inside_polygon(polygon):
                        cells_at_boundary_inside.append(cell)
                        break
            else:
                for neighbor in cell.get_neighbors():
                    if neighbor.is_inside_polygon(polygon):
                        cells_at_boundary_outside.append(cell)
                        break                

        return cells_at_boundary_inside, cells_at_boundary_outside



