from collections.abc import Callable
from dolfinx import fem, mesh, la
import basix.ufl
import ufl
from mpi4py import MPI
import numpy as np
import scipy.sparse
from plotter import visualize_mixed, loglog_plot, convergence_rate_plot



def boundary_functions_factory(neumann_boundary: str = "bottom") -> tuple[Callable[[np.ndarray], np.ndarray], 
                                                                        Callable[[np.ndarray], np.ndarray]]:
    """
    Factory function to create functions for identifying Dirichlet and Neumann boundaries
    based on the boundary name.
    
    Returns vectorized Neumann and Dirichlet boundary tag functions.
    
    """
    all_boundary_conditions = {'left' : lambda x : np.isclose(x[0], 0.0), 
                               'bottom' : lambda x : np.isclose(x[1], 0.0), 
                               'right' : lambda x: np.isclose(x[0], 1.0), 
                               'top' : lambda x : np.isclose(x[1], 1.0)
    }
    on_neumann = all_boundary_conditions.pop(neumann_boundary)
    def on_dirichlet(x):
        return np.logical_or.reduce([func(x) for func in all_boundary_conditions.values()])
        
    return on_dirichlet, on_neumann  
    
def u_exact_numpy(x: np.ndarray) -> np.ndarray:
    """
    Exact solution for the velocity field.
    Used for interpolating onto the function space.
    
    """
    return np.sin(np.pi * x[1]), np.cos(np.pi * x[0])

def p_exact_numpy(x: np.ndarray) -> np.ndarray:
    """
    Exact solution for the pressure field.
    Used for interpolating onto the function space.
    
    """
    return np.sin(2*np.pi * x[0])

def u_exact_ufl(x: ufl.SpatialCoordinate) -> ufl.Coefficient:
    """
    Exact solution for the velocity field.
    Used for symbolicaly defining the exact solution, in the residual form of the problem.
    
    """
    return ufl.as_vector([ufl.sin(ufl.pi * x[1]), ufl.cos(ufl.pi * x[0])])


def p_exact_ufl(x: ufl.SpatialCoordinate) -> ufl.Coefficient:
    """
    Exact solution for the pressure field.
    Used for symbolicaly defining the exact solution, in the residual form of the problem.
    
    """
    return ufl.sin(2 * ufl.pi * x[0])


def setup_system(W: fem.FunctionSpace, enforce_neumann=True):
    """
    Setup the variational problem for the Stokes equations. The math of the weak form is defined here.
    
    """
    domain = W.mesh
    
    u, p = ufl.TrialFunctions(W) #linear combs of basis functions
    v, q = ufl.TestFunctions(W)

    x = ufl.SpatialCoordinate(domain)
    
    
    f = -ufl.div(ufl.grad(u_exact_ufl(x))) - ufl.grad(p_exact_ufl(x))
    F = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    F += ufl.inner(p, ufl.div(v)) * ufl.dx
    F += ufl.inner(ufl.div(u), q) * ufl.dx
    F -= ufl.inner(f, v) * ufl.dx
    
    if enforce_neumann: 
        n = ufl.FacetNormal(domain)
        h = (ufl.grad(u_exact_ufl(x)) + p_exact_ufl(x)*ufl.Identity(len(u_exact_ufl(x)))) * n
        F -= ufl.inner(h, v) * ufl.ds
        #we can integrate over all of ds since v=0 on the dirichlet boundary, which is the remainding part of the boundary
    else:
        pass #Do nothing boundary condtion
    
    a, L = ufl.system(F)
    
    return a, L


def solve_stokes(N, polypair, neumann_boundary ="right", enforce_neumann=False, plot=False, savefig=False, savename=""):
    """
    Solve the Stokes problem on a unit square using mixed finite elements.
    
    Parameters:
    N : int
        Number of cells in each direction.
    polypair : tuple
        Polynomial degree pair for the velocity and pressure spaces.
    neumann_boundary : str
        Name of the Neumann boundary.
    enforce_neumann : bool
        Whether to enforce the Neumann boundary condition.
    .
    .
    Returns:
    uh : dolfinx.fem.Function
        Approximate velocity field.
    ph : dolfinx.fem.Function
        Approximate pressure field.
    u_exact : dolfinx.fem.Function
        Exact velocity field. 
    p_exact : dolfinx.fem.Function
        Exact pressure field.
    
    """
        
    on_dirichlet, on_neumann = boundary_functions_factory(neumann_boundary)

    #-----setup domain and function space----
    p_u, p_p = polypair
    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    el_u = basix.ufl.element("Lagrange", domain.basix_cell(), p_u, shape=(domain.geometry.dim,))
    el_p = basix.ufl.element("Lagrange", domain.basix_cell(), p_p)
    el_mixed = basix.ufl.mixed_element([el_u, el_p])    
    W = fem.functionspace(domain, el_mixed)
    
    #-----setup variational problem----
    a, L = setup_system(W, enforce_neumann)


    #------Dirichlet boundary conditions----
    W0 = W.sub(0)
    V, V_to_W0 = W0.collapse()
    
    u_exact = fem.Function(V)
    u_exact.interpolate(u_exact_numpy)

    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    dir_facets = mesh.locate_entities_boundary(domain, domain.topology.dim - 1, on_dirichlet)
    combined_dofs = fem.locate_dofs_topological((W0, V), domain.topology.dim - 1, dir_facets)
    bc = fem.dirichletbc(u_exact, combined_dofs, W0)
    bcs = [bc]
  
  
    #------Form, create, assemble and solve linear system----
    a_compiled = fem.form(a) #create c code for the form
    L_compiled = fem.form(L)
    A = fem.create_matrix(a_compiled) #create matrix code for the form
    b = fem.create_vector(L_compiled)
    A_scipy = A.to_scipy()
    fem.assemble_matrix(A, a_compiled, bcs=bcs) #actually fill the matrix
    fem.assemble_vector(b.array, L_compiled)
    fem.apply_lifting(b.array, [a_compiled], [bcs]) 
    b.scatter_reverse(la.InsertMode.add) #tell ghosted vector to add values to local dofs
    bc.set(b.array) #set the boundary condition values to the vector

    A_inv = scipy.sparse.linalg.splu(A_scipy) #lu factorization
    
    wh = fem.Function(W)# solve Ax=b for x
    wh.x.array[:] = A_inv.solve(b.array) #put the solution in wh

    
    return wh


def raised_difference(exact: Callable[[np.ndarray], np.ndarray], approx: fem.Function, degree_raise: int = 3) -> fem.Function:  
    """ Get exact solution for the same quadrature points as approximation, interpolate both to a higher 
            space and return the difference between the exact and approximate solutions.

    Based on https://jsdokken.com/dolfinx-tutorial/chapter4/convergence.html with help from August Femtehjell

    Args:
        exact (Callable[[ufl.SpatialCoordinate], ufl.Coefficient]): The exact solution.
        approx (dolfinx.fem.Function): The approximate solution.
        degree_raise (int, optional): The degree raise for the space. Defaults to 3.

    Returns:
        dolfinx.fem.Function: The error function.
    """
    # Get the function space and mesh   
    V = approx.function_space
    domain = V.mesh
    degree = V.ufl_element().degree
    family = V.ufl_element().family_name
    shape = V.value_shape
 
    # Create a higher-order function space
    Ve = fem.functionspace(domain, (family, degree + degree_raise, shape))
    
    # Interpolate the exact solution to the higher-order function space
    u_ex = fem.Function(Ve)
    u_ex.interpolate(exact)

    # Interpolate the approximate solution to the higher-order function space
    u_h = fem.Function(Ve)
    u_h.interpolate(approx)

    difference = fem.Function(Ve)
    # Compute the difference between the exact and approximate solutions
    difference.x.array[:] = u_ex.x.array - u_h.x.array

    return difference


def L2_error(exact, approx, comm=None, measure=ufl.dx):
    """
    Calculate the L2 error between the exact and approximate solutions.
    
    """
    if comm is None:
        comm = approx.function_space.mesh.comm
    diff = raised_difference(exact, approx)
    # Define the error form
    error_form = fem.form(ufl.inner(diff, diff) * measure)
    # Assemble the error and perform a global reduction
    error_value = np.sqrt(comm.allreduce(fem.assemble_scalar(error_form), op=MPI.SUM))
    return error_value


def H1_seminorm_error(exact, approx, measure=ufl.dx):
    """
    Calculate the H1 error (gradient error) between the exact and approximate solutions.
    returns:
    error_value : float
        ||grad(exact - approx)||_L2
    """
    comm = approx.function_space.mesh.comm
    diff = raised_difference(exact, approx)
    error_form = fem.form(ufl.inner(ufl.grad(diff), ufl.grad(diff)) * measure)
    error_value = np.sqrt(comm.allreduce(fem.assemble_scalar(error_form), op=MPI.SUM))
    return error_value


def calculate_shear_stress(u_exact, uh, neumann_boundary="left"):
    """
    Calculate the L2 error in the shear stress on a specified Neumann boundary.
    
    returns:
    shear_error : float
        ||grad(u_exact) * t - grad(uh) * t||_L2 #could also have done H1 seminorm of u*t,uh*t

    """
    comm = uh.function_space.mesh.comm
    _, on_neumann = boundary_functions_factory(neumann_boundary)
    # Locate facets on the Neumann boundary
    neumann_facets = mesh.locate_entities_boundary(
        uh.function_space.mesh,
        uh.function_space.mesh.topology.dim - 1,
        on_neumann
    )
    # Create meshtags for the Neumann boundary; here, all facets are tagged with 0
    mt = mesh.meshtags(
        uh.function_space.mesh,
        uh.function_space.mesh.topology.dim - 1,
        neumann_facets,
        np.full(len(neumann_facets), 0, dtype=np.int32)
    )
    ds = ufl.Measure("ds", domain=uh.function_space.mesh, subdomain_data=mt)
    # Define normal and tangent vectors
    n = ufl.FacetNormal(uh.function_space.mesh)
    t = ufl.as_vector([n[1], -n[0]])
    
    diff = raised_difference(u_exact, uh)
    error_form = fem.form(ufl.inner(ufl.grad(diff)*t, ufl.grad(diff)*t) * ds(0))
    error_value = np.sqrt(comm.allreduce(fem.assemble_scalar(error_form), op=MPI.SUM))
    return error_value


def experiment_ex66(Ns):
    """
    Run experiment ex66 to evaluate the convergence of the solution for various polynomial pairs.
    
    """
    polypairs = [(4, 3), (4, 2), (3, 2), (3, 1)]
    num_N = len(Ns)
    num_polypairs = len(polypairs)
    Es = np.zeros((num_N, num_polypairs))
    Hs = np.zeros((num_N, num_polypairs))
    
    for j, polypair in enumerate(polypairs):
        for i, N in enumerate(Ns):
            wh = solve_stokes(N, polypair)
            # Compute errors (order: exact, approx)
            uh = wh.sub(0).collapse()
            ph = wh.sub(1).collapse()
            
            error_pressure = L2_error(p_exact_numpy, ph)
            error_velocity = H1_seminorm_error(u_exact_numpy, uh)
            Es[i, j] = error_pressure + error_velocity
            Hs[i, j] = 1.0 / N
    
    # Compute convergence rates between successive mesh refinements
    rates = np.log(Es[:-1] / Es[1:]) / np.log(Hs[:-1] / Hs[1:])
    mean_rates = np.mean(rates, axis=0)
    print(f"Mean error convergence rates of solution: {mean_rates}")
    return Es, Hs, rates


def experiment_ex67(Ns):
    """
    Run experiment ex67 to evaluate the convergence of both the solution and the shear stress on a Neumann boundary.

    """
    neumann_boundary = "left"
    polypair = (3, 2)
    num_N = len(Ns)
    Es = np.zeros((num_N, 2))  # Column 0: solution error; Column 1: shear stress error
    Hs = np.zeros(num_N)
    
    for i, N in enumerate(Ns):
        wh = solve_stokes(
            N, polypair, enforce_neumann=True, neumann_boundary=neumann_boundary
        )
        uh = wh.sub(0).collapse()
        ph = wh.sub(1).collapse()
        
        error_solution = L2_error(p_exact_numpy, ph) + H1_seminorm_error(u_exact_numpy, uh) + L2_error(u_exact_numpy, uh)
        error_shear = calculate_shear_stress(u_exact_numpy, uh, neumann_boundary=neumann_boundary)
        Es[i, 0] = error_solution
        Es[i, 1] = error_shear
        Hs[i] = 1.0 / N
    
    rates_sol = np.log(Es[:-1, 0] / Es[1:, 0]) / np.log(Hs[:-1] / Hs[1:])
    rates_shear = np.log(Es[:-1, 1] / Es[1:, 1]) / np.log(Hs[:-1] / Hs[1:])
    return Es, Hs, rates_sol, rates_shear


def test(plot=False, enforce_neumann=True, neumann_boundary='right', savefig=False, savename=""):
    """
    Test the Stokes solver and error calculations, and optionally plot or save the results.
  
    """
    wh = solve_stokes(
        10, (3, 2), plot=plot, enforce_neumann=enforce_neumann,
        neumann_boundary=neumann_boundary, savefig=savefig, savename=savename
    )
    if plot:
        visualize_mixed(wh, scale=0.1, savefig=savefig, savename=savename)

    uh = wh.sub(0).collapse()
    ph = wh.sub(1).collapse()
    
    print(uh.function_space.mesh)
    # Compute errors with the convention: exact, approx
    error = H1_seminorm_error(u_exact_numpy, uh) + L2_error(p_exact_numpy, ph)
    comm = uh.function_space.mesh.comm
    shear_error = calculate_shear_stress(u_exact_numpy, uh, neumann_boundary='left')
    if comm.rank == 0:
        print(f"Error: {error:.2e}")
        print(f"Shear stress error: {shear_error:.2e}")

if __name__ == "__main__":
    Ns = [2, 4, 8, 16, 32, 64]
    #test(plot=True, savefig=False, savename='66', neumann_boundary="right") #plot
    #test(plot=True, enforce_neumann=True, neumann_boundary='right', savefig=False, savename='67') #plot
    #test(plot=True, enforce_neumann=True, neumann_boundary='bottom', savefig=False, savename='67') #plot
    
    #quit()
    Es66, Hs66, rates66 = experiment_ex66(Ns)
    Es67, Hs67, rates_sol67, rates_shear67 = experiment_ex67(Ns)
    
    if MPI.COMM_WORLD.rank == 0:
        print(f"Mean error convergence rates of solutions ex66: {np.mean(rates66, axis=1)}")
        print(f"Mean error convergence rates of solution ex67: {np.mean(rates_sol67)}")
        print(f"Mean error convergence rates of shear stress ex67: {np.mean(rates_shear67)}")

        loglog_plot(Hs67, Es67, Hs66, Es66)
        convergence_rate_plot(Ns, rates66, rates_sol67, rates_shear67)