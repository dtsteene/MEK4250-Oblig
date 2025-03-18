import pyvista
import dolfinx
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def visualize_mixed(mixed_function: dolfinx.fem.Function, scale=1.0, savefig=False, savename=""):
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
        
    
def loglog_plot(Hs67, Es67, Hs66, Es66):
    """
    Plot log-log error plots for the two different boundary conditions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Log-Log Error plots", fontsize=16)

    axes[0].set_title("Neumann BC on Bottom Wall P3-P2")
    axes[0].loglog(Hs67, Es67[:, 0], label="Error solution")
    axes[0].loglog(Hs67, Es67[:, 1], label="Error shear stress")
    axes[0].set_xlabel("h")
    axes[0].set_ylabel("Error")
    axes[0].legend()

    Hs_same = Hs66[:, 0]
    axes[1].set_title("Neumann BC (do nothing) on Right Wall")
    axes[1].loglog(Hs_same, Es66[:, 0], label="Error P4-P3")
    axes[1].loglog(Hs_same, Es66[:, 1], label="Error P4-P2")
    axes[1].loglog(Hs_same, Es66[:, 2], label="Error P3-P2")
    axes[1].loglog(Hs_same, Es66[:, 3], label="Error P3-P1")
    axes[1].set_xlabel("h")
    axes[1].set_ylabel("Error")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("figs/loglog.png", dpi=300)
    plt.show()
    plt.close()
    

def convergence_rate_plot(Ns, rates66, rates_sol67, rates_shear67):
    """
    Plot convergence rates for the two different boundary conditions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Convergence Rates", fontsize=16)
    
    x = np.array(Ns[:-1])
    
    axes[0].plot(Ns[:-1], rates66[:, 0], label="P4-P3", color="blue")
    axes[0].axhline(y=4, color="blue", linestyle="--")
    
    axes[0].plot(Ns[:-1], rates66[:, 1], label="P4-P2", color="orange")
    axes[0].axhline(y=3, linestyle="--", color="orange")
    
    axes[0].plot(Ns[:-1], rates66[:, 2], label="P3-P2")
    axes[0].plot(Ns[:-1], rates66[:, 3], label="P3-P1", color="red")
    axes[0].axhline(y=2, linestyle="--", color="red")
    axes[0].set_ylabel("Convergence rate")
    axes[0].set_xlabel(r"$N$")
    axes[0].legend()
    axes[0].set_title("Do Nothing BC on Right Wall")

    axes[1].plot(Ns[:-1], rates_sol67, label="Bottom BC")
    axes[1].plot(Ns[:-1], rates_shear67, label="Shear Stress Left Wall")
    axes[1].axhline(y=3, linestyle="--")
    axes[1].set_ylabel("Convergence rate")
    axes[1].set_xlabel(r"$N$")
    axes[1].legend()
    axes[1].set_title("Neumann BC on Bottom Wall P3-P2")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("figs/convergence_rates.png", dpi=300)
    plt.show()
    plt.close()

    