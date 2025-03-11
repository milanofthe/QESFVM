###################################################################################
##
##                           Utility Functions
##
##                          Milan Rother 2023/24
##
###################################################################################


# IMPORTS =========================================================================

import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
import logging


# FUNCTIONS ============================================================================

def qt_rl(d_max, d_min):
    """
    function that determines the required number of quadtree 
    refinement levels to reach a certain minimum resolution
    """
    if d_max <= d_min:
        return 0
    else:
        return int(np.ceil(np.log2(d_max / d_min)))



# INTERPOLATION ========================================================================

def bivariate(points, n):

    """
    construct n-th order bivariate (x,y) polynomial interpolation least squares matrix from a given set of points
    """

    M = np.array([[(x**i)*(y**j) for i in range(n+1) for j in range(n+1)] for x, y in points])

    A = np.dot(M.T, M)

    # print(np.linalg.det(A))
    return np.linalg.solve(A, M.T)


def construct_surface_integral_E(points, n, x_max, x_min, y_max, y_min):
    P = np.array([ i/(j+1) * x_max**(i-1) * (y_max**(j+1) - y_min**(j+1)) for i in range(n+1) for j in range(n+1)])
    return np.dot(bivariate(points, n).T, P).flatten()


def construct_surface_integral_W(points, n, x_max, x_min, y_max, y_min):
    P = np.array([ -i/(j+1) * x_min**(i-1) * (y_max**(j+1) - y_min**(j+1)) for i in range(n+1) for j in range(n+1)])
    return np.dot(bivariate(points, n).T, P).flatten()


def construct_surface_integral_N(points, n, x_max, x_min, y_max, y_min):
    P = np.array([ j/(i+1) * (x_max**(i+1) - x_min**(i+1)) * y_max**(j-1) for i in range(n+1) for j in range(n+1)])
    return np.dot(bivariate(points, n).T, P).flatten()


def construct_surface_integral_S(points, n, x_max, x_min, y_max, y_min):
    P = np.array([ -j/(i+1) * (x_max**(i+1) - x_min**(i+1)) * y_min**(j-1) for i in range(n+1) for j in range(n+1)])
    return np.dot(bivariate(points, n).T, P).flatten()





# PLOTTING =============================================================================

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):

    """
    Add a vertical color bar to an image plot.
    """
    
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)



# LOGGING ==============================================================================

def configure_logging(filename=None):
    """
    setup logging configuration

    for saving to *.log file, specify filename accordinngly
    """
    logging.basicConfig(level=logging.INFO,  # Adjust level as needed
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=filename,  # Log to a file, remove to log to console
                        filemode="w")  # 'w' for overwrite, 'a' for append


def timer(func):
    """
    shows the execution time in milliseconds
    of the function object passed
    """
    def wrap_func(*args, **kwargs):
        t1 = perf_counter()
        result = func(*args, **kwargs)
        t2 = perf_counter()
        print(f'Function {func.__name__!r} executed in {(t2-t1)*1e3:.2f}ms')
        return result
    return wrap_func


def performance_enumerate(iterator, max_logs=10, log=True):
    """
    Generator that replaces range and adds timing and progress logging if log mode is enabled.
    The function ensures that the progress is logged up to 'max_logs' times throughout the iteration.
    
    INPUTS :
        start    : Start of the range
        stop     : End of the range
        step     : Step size between each iteration
        max_logs : Maximum number of progress logs in percent '%'
        log      : Flag to enable log mode with logging
    """
    total_steps = len(iterator)

    # Calculate log interval to not exceed max_logs
    log_interval = max(1, total_steps // max_logs)  

    if log:
        start_time = perf_counter()
        logging.info(f"starting - total: {total_steps}")

    for i, value in enumerate(iterator):
        if log and ((i+1) % log_interval == 0 or (i+1) == total_steps):
            percent_complete = ((i+1) / total_steps) * 100
            logging.info(f"progress - completed: {percent_complete:.2f}%")

        yield i, value

    if log:
        runtime = perf_counter() - start_time
        logging.info(f"complete - runtime: {runtime*1e3:.2f}ms")