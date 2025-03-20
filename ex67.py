from collections.abc import Callable
from dolfinx import fem, mesh
import basix.ufl
import ufl
from mpi4py import MPI
import numpy as np
from plotter import plot_shear_stress_rates, plot_convergence_rates
from dolfinx.fem.petsc import LinearProblem



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

    dir_facets = mesh.locate_entities_boundary(domain, domain.topology.dim - 1, on_dirichlet)
    combined_dofs = fem.locate_dofs_topological((W0, V), domain.topology.dim - 1, dir_facets)
    bc = fem.dirichletbc(u_exact, combined_dofs, W0)
    bcs = [bc]
  
    #-----solve the problem----
    problem = LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )
    wh = problem.solve()
    
    return wh

    


def raised_difference(exact: Callable[[np.ndarray], np.ndarray], 
                      approx: fem.Function, 
                      degree_raise: int = 3) -> fem.Function:  
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


def L2_error(exact: Callable[[np.ndarray], np.ndarray], approx: fem.Function, comm=None, measure=ufl.dx):
    """
    Calculate the L2 error between the exact and approximate solutions.
    
    """
    if comm is None:
        comm = approx.function_space.mesh.comm
    diff = raised_difference(exact, approx)
    error_form = fem.form(ufl.inner(diff, diff) * measure)
    error_value = np.sqrt(comm.allreduce(fem.assemble_scalar(error_form), op=MPI.SUM))
    return error_value


def H1_seminorm_error(exact: Callable[[np.ndarray], np.ndarray], approx: fem.Function, measure=ufl.dx):
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


def calculate_shear_stress(u_exact: Callable[[np.ndarray], np.ndarray], uh: fem.Function, neumann_boundary="left"):
    """
    Calculate the L2 error in the shear stress on a specified Neumann boundary.
    
    returns:
    shear_error : float
        ||grad(u_exact - uh) * t||_L2 

    """
   
    _, on_neumann = boundary_functions_factory(neumann_boundary)
    
    domain = uh.function_space.mesh
    neumann_facets = mesh.locate_entities_boundary(
        domain,
        domain.topology.dim - 1,
        on_neumann
    )
    
    mt = mesh.meshtags(
        domain,
        domain.topology.dim - 1,
        neumann_facets,
        np.full_like(neumann_facets, 0, dtype=np.int32)
    )
    
    ds = ufl.Measure("ds", domain=uh.function_space.mesh, subdomain_data=mt)
    # Define normal and tangent vectors
    diff = raised_difference(u_exact, uh)
    
    n = ufl.FacetNormal(diff.function_space.mesh)
    t = ufl.as_vector([n[1], -n[0]])
    
    
    error_form = fem.form(ufl.inner(ufl.grad(diff)*t, ufl.grad(diff)*t) * ds(0))
    comm = uh.function_space.mesh.comm
    error_value = np.sqrt(comm.allreduce(fem.assemble_scalar(error_form), op=MPI.SUM))
    return error_value


        
def experiment_poly_pairs(Ns: list[int], polypairs: list[tuple[int, int]], 
                            enforce_neumann: bool = False, neumann_boundary: str = "right"):
    """
    Run experiments for a set of polynomial pairs.
    
    If enforce_neumann is False, it runs the standard (do nothing) experiment,
    computing a single error (pressure + velocity error).
    If enforce_neumann is True, it runs the experiment with a Neumann boundary condition
    (using the specified neumann_boundary) and computes two errors:
      - error_solution: combined solution error (including an extra L2 velocity error)
      - error_shear: error in the computed shear stress.
    
    Returns:
      For do nothing: (Es, Hs, rates)
      For Neumann: (Es_solution, Es_shear, Hs, rates_solution, rates_shear)
    """
    num_N = len(Ns)
    num_poly = len(polypairs)
    Es = np.zeros((num_N, num_poly))
    Hs = np.zeros((num_N, num_poly))
    
    if enforce_neumann:
        Es_shear = np.zeros((num_N, num_poly))
    
    for j, poly in enumerate(polypairs):
        for i, N in enumerate(Ns):
            if enforce_neumann:
                wh = solve_stokes(N, poly, enforce_neumann=True, neumann_boundary=neumann_boundary)
                uh = wh.sub(0).collapse()
                ph = wh.sub(1).collapse()
                error_solution = (L2_error(p_exact_numpy, ph) +
                                  H1_seminorm_error(u_exact_numpy, uh) +
                                  L2_error(u_exact_numpy, uh))
                error_shear = calculate_shear_stress(u_exact_numpy, uh, neumann_boundary=neumann_boundary)
                Es[i, j] = error_solution
                Es_shear[i, j] = error_shear
            else:
                wh = solve_stokes(N, poly)
                uh = wh.sub(0).collapse()
                ph = wh.sub(1).collapse()
                error = L2_error(p_exact_numpy, ph) + H1_seminorm_error(u_exact_numpy, uh) + L2_error(u_exact_numpy, uh)
                Es[i, j] = error
            Hs[i, j] = 1.0 / N


    if enforce_neumann:
        rates_solution = np.log(Es[:-1, :] / Es[1:, :]) / np.log(Hs[:-1, :] / Hs[1:, :])
        rates_shear = np.log(Es_shear[:-1, :] / Es_shear[1:, :]) / np.log(Hs[:-1, :] / Hs[1:, :])
        return Es, Es_shear, Hs, rates_solution, rates_shear
    else:
        rates = np.log(Es[:-1, :] / Es[1:, :]) / np.log(Hs[:-1, :] / Hs[1:, :])
        return Es, Hs, rates

if __name__ == "__main__":
    Ns = [4, 8, 16, 32, 64, 128]
    polypairs = [(4, 3), (4, 2), (3, 2), (3, 1)]
    
    # Ex 6.6
    Es_dn, Hs_dn, rates_dn = experiment_poly_pairs(Ns, polypairs, enforce_neumann=False)
    
    # Ex 6.7
    Es_neu, Es_shear_neu, Hs_neu, rates_sol_neu, rates_shear_neu = experiment_poly_pairs(
        Ns, polypairs, enforce_neumann=True, neumann_boundary="bottom"
    )
    
    if MPI.COMM_WORLD.rank == 0:
    
        print("Mean convergence rates for Do Nothing BC:")
        print(np.mean(rates_dn, axis=0))
        print("Mean convergence rates for Neumann BC (Solution):")
        print(np.mean(rates_sol_neu, axis=0))
        print("Mean convergence rates for Neumann BC (Shear Stress):")
        print(np.mean(rates_shear_neu, axis=0))
        
    
        plot_convergence_rates(Ns, rates_dn, rates_sol_neu, polypairs)
        plot_shear_stress_rates(Ns, rates_shear_neu, polypairs)
    