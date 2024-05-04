# %%
import matplotlib.pyplot as plt
import pyvista
import ufl
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI
import dolfinx
from dolfinx import fem, mesh, io, plot
from dolfinx.fem import Constant, Function, FunctionSpace, assemble_scalar, dirichletbc, form, locate_dofs_geometrical
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
from dolfinx.io import VTXWriter
from dolfinx.mesh import create_unit_square
from dolfinx.plot import vtk_mesh
from ufl import (FacetNormal, FiniteElement, Identity, TestFunction, TrialFunction, VectorElement,
                 grad, div, dot, ds, dx, inner, lhs, nabla_grad, rhs, sym)

import dolfinx.fem.petsc
import dolfinx.nls.petsc


# Define temporal parameters
t = 0  # Start time
T = 2.0  # Final time
T = 8.0  # Final time
dt = 1e-3  # Time step size

# Define the domain
n_x = 2**7 + 1
domain = mesh.create_interval(comm=MPI.COMM_WORLD, points=(0.0, 1.0), nx=n_x)

# Define function space (P2 element for better accuracy)
V = fem.FunctionSpace(domain, ("Lagrange", 1))

# Define initial condition function
def initial_condition(x, mu):
    return mu * np.sin(2*np.pi*x[0]) * (x[0] <= 0.5)

# Create boundary condition
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

# Define parameters
Reynolds_nominal = int(1e3)
# Reynolds_lst = [Reynolds_nominal, 1e3 * 1.01, 1e3 * 0.99, 1e3 * 1.05, 1e3 * 0.95]
# Reynolds_lst = [Reynolds_nominal * 1.10, Reynolds_nominal * 0.90, 1e3 * 1.20, Reynolds_nominal * 0.80]
Reynolds_lst = [Reynolds_nominal * 1.15, Reynolds_nominal * 0.85, 1e3 * 1.25, Reynolds_nominal * 0.75]
# Reynolds_lst = [Reynolds_nominal]

# Names = [f"RE{Reynolds_nominal}", f"RE{Reynolds_nominal}_p1per", 
#          f"RE{Reynolds_nominal}_n1per", f"RE{Reynolds_nominal}_p5per", f"RE{Reynolds_nominal}_n5per"]
# Names = [f"RE{Reynolds_nominal}_p10per", 
#          f"RE{Reynolds_nominal}_n10per", f"RE{Reynolds_nominal}_p20per", f"RE{Reynolds_nominal}_n20per"]
Names = [f"RE{Reynolds_nominal}_p15per", 
         f"RE{Reynolds_nominal}_n15per", f"RE{Reynolds_nominal}_p25per", f"RE{Reynolds_nominal}_n25per"]

# Names = [f"RE{Reynolds_nominal}"]

# Advective flux of conservation law:
e0 = PETSc.ScalarType(1.0)
def flux(u):
    return 0.5 * u * u * e0

# Loop for solving with multiple mu values
mu_values = [1.1, 1.05, 1, 0.95, 0.9]

for i in range(len(Reynolds_lst)):
    Reynolds = Reynolds_lst[i]
    solutions = []
    gradient_lst = []
    for mu in mu_values:
        
        sol_t = []
        grad_t = []
        
        # Create initial condition
        uh = fem.Function(V)
        uh.name = "uh"
        uh.interpolate(lambda x: initial_condition(x, mu))
        
        u_n = fem.Function(V)  # Trial function
        u_n.name = "u_n"
        u_n.interpolate(lambda x: initial_condition(x, mu))

        # Define the variational problem
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        f = fem.Constant(domain, PETSc.ScalarType(0))
        F = (dot(uh - u_n, v) / dt
            + dot(uh * uh.dx(0), v)
            + 1/Reynolds * dot(grad(uh), grad(v))) * dx

        # Define the solver
        problem = fem.petsc.NonlinearProblem(F, uh, bcs=[bc])
        solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
        solver.convergence_criterion = "incremental"
        solver.rtol = 1e-6
        solver.report = True
        
        # Time-stepping loop
        t = 0.0
        while t < T:
            t += dt
            # Solve linear problem
            solver.solve(uh)
            uh.x.scatter_forward()
            
            # update solution
            u_n.x.array[:] = uh.x.array
            
            gradient = ufl.grad(uh)
            grad_uh = fem.FunctionSpace(domain, ("Lagrange", 1))
            grad_uh_expr = fem.Expression(gradient, grad_uh.element.interpolation_points())
            gradients = fem.Function(grad_uh)
            gradients.interpolate(grad_uh_expr)
            
            
            sol_t.append(u_n.vector[:])
            grad_t.append(gradients.vector[:])

        # Append solution for this mu value
        solutions.append(np.array(sol_t))
        gradient_lst.append(np.array(grad_t))
        
        print("Solution for mu = {} done".format(mu))
        print("Shape of solution: ", np.array(sol_t).shape)

    # Get the solutions as a NumPy array
    u_sol_all = np.array(solutions)
    grad_all = np.array(gradient_lst)

    # save the solutions to a numpy file
    np.save(f"burgersFEniCSx_u_sol_all_{Names[i]}.npy", u_sol_all)
    # np.save("burgersFEniCSx_grad_all_RE1000npy", grad_all)
