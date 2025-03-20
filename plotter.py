import pyvista
import dolfinx
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def visualize_mixed(mixed_function: dolfinx.fem.Function, scale=0.1, savefig=False, savename=""):
    """
    Plot a mixed function with a vector and scalar component. Mostly
    compied from dokken tutorial.
    
    """
    u_c = mixed_function.sub(0).collapse()
    p_c = mixed_function.sub(1).collapse()

    u_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(u_c.function_space))

    # Pad u to be 3D
    gdim = u_c.function_space.mesh.geometry.dim
    assert len(u_c) == gdim
    u_values = np.zeros((len(u_c.x.array) // gdim, 3), dtype=np.float64)
    u_values[:, :gdim] = u_c.x.array.real.reshape((-1, gdim))

    # Create a point cloud of glyphs
    u_grid["u"] = u_values
    glyphs = u_grid.glyph(orient="u", factor=scale)
    pyvista.set_jupyter_backend("static")
    plotter = pyvista.Plotter()
    plotter.add_key_event("Escape", lambda: plotter.close())
    plotter.add_mesh(u_grid, show_edges=False, show_scalar_bar=False)
    plotter.add_mesh(glyphs)
    plotter.view_xy()
    plotter.show()
    if savefig:
        #check if figs folder exists and create it if not
        folder_path = Path("figs")
        folder_path.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(r"figs/velocity_" + savename + ".png", transparent_background=True)

    p_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(p_c.function_space))
    p_grid.point_data["p"] = p_c.x.array
    plotter_p = pyvista.Plotter()
    plotter_p.add_mesh(p_grid, show_edges=False)
    plotter_p.view_xy()
    plotter_p.show()
    if savefig:
        plotter_p.screenshot(r"figs/pressure_" + savename + ".png", transparent_background=True)
        
    



def plot_convergence_rates(Ns, rates_dn, rates_sol_neu, polypairs):
    """
    Plot convergence rates (solution error) for the two different boundary conditions
    in a single figure with two subplots.
    
    Left subplot: Do Nothing BC.
    Right subplot: Neumann BC (bottom) solution error.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(r"Error Convergence Rates $||u - u_h||_1 + || p - p_h||_0$ ", fontsize=16)
    x_ticks = [f"$N_{{{i+1}}}/N_{{{i}}}$" for i in range(len(Ns) - 1)]
    x = range(len(x_ticks))
    
    # Plot for Do Nothing BC.
    for j, poly in enumerate(polypairs):
        axes[0].plot(x, rates_dn[:, j], marker='o', label=f"P{poly[0]}-P{poly[1]}")
    axes[0].set_title("Do Nothing Neumann BC")
    axes[0].set_xlabel("Mesh Refinement")
    axes[0].set_ylabel("Convergence Rate")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(x_ticks)
    axes[0].legend()
    
    # Plot for Neumann BC (Bottom) - solution error.
    for j, poly in enumerate(polypairs):
        axes[1].plot(x, rates_sol_neu[:, j], marker='o', label=f"P{poly[0]}-P{poly[1]}")
    axes[1].set_title("Neumann BC (Bottom)")
    axes[1].set_xlabel("Mesh Refinement")
    axes[1].set_ylabel("Convergence Rate")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(x_ticks)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig("figs/convergence_rates_subplot.png", dpi=300)
    plt.show()
    plt.close()


def plot_shear_stress_rates(Ns, rates_shear_neu, polypairs):
    """
    Plot shear stress convergence rates for the Neumann BC (Bottom)
    for all polynomial pairs.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    x_ticks = [f"$N_{{{i+1}}}/N_{{{i}}}$" for i in range(len(Ns) - 1)]
    x = range(len(x_ticks))
    
    for j, poly in enumerate(polypairs):
        ax.plot(x, rates_shear_neu[:, j], marker='o', label=f"P{poly[0]}-P{poly[1]}")
    ax.set_title("Neumann BC (Bottom) - Shear Stress Convergence Rates")
    ax.set_xlabel("Mesh Refinement")
    ax.set_ylabel("Convergence Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(x_ticks)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("figs/shear_stress_convergence_rates.png", dpi=300)
    plt.show()
    plt.close()

    