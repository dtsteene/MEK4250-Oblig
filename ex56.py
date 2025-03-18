from dolfinx import mesh, fem
from dolfinx.fem import petsc
import ufl
from mpi4py import MPI
import numpy as np
from scipy.linalg import eig
from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.pyplot as plt

N = 300
left = 0.0
right = 1.0
domain = mesh.create_interval(MPI.COMM_WORLD, N, [left, right])
V = fem.functionspace(domain, ("Lagrange", 1))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

alpha = 1.0e-5

x = ufl.SpatialCoordinate(domain)[0]

a = (x * u.dx(0) * v + alpha * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
m = u * v * ufl.dx

def boundary(x):
    return np.logical_or(np.isclose(x[0], left), np.isclose(x[0],
    right))

boundary_dofs = fem.locate_dofs_geometrical(V, boundary)
bc = fem.dirichletbc(0.0, boundary_dofs, V)


A = petsc.assemble_matrix(fem.form(a), bcs=[bc])
A.assemble()
M = petsc.assemble_matrix(fem.form(m), bcs=[bc])
M.assemble()

Ai, Aj, Av = A.getValuesCSR()
Mi, Mj, Mv = M.getValuesCSR()
A_dense = csr_matrix((Av, Aj, Ai)).toarray()
M_dense = csr_matrix((Mv, Mj, Mi)).toarray()
evals, _ = eig(A_dense, M_dense)
plt.plot(evals.real, evals.imag, "o")
plt.show()

finite_evals = evals[np.isfinite(evals)]
real_positive = [float((np.real(ev))) for ev in finite_evals if
np.isreal(ev) and ev.real > 0]
print("Number of positive real eigenvalues:", len(real_positive))
print("Positive real eigenvalues:", real_positive)