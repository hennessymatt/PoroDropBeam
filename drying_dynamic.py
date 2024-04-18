from dolfin import *
from multiphenics import *
from helpers import *
import numpy as np
import json
from pathlib import Path

meshname = 'parabolic_0p01_0p1_fine'
fname = 'dynamic/' + meshname + '/Pe_huge/E_one/'
# fname = 'dynamic/' + meshname + '/nice_pic/'

epsilon = 0.01
delta = 0.1
Q = 5/epsilon
E = 1
# E = delta / epsilon

"""
    parameters
"""

# physical parameters
E_b = 1 / delta**2 / E

nu_d = 0.2
nu_b = 0.265

a_d = nu_d / (1 + nu_d) / (1 - 2 * nu_d)
b_d = 1 / 2 / (1 + nu_d)

a_b = E_b * nu_b / (1 + nu_b) / (1 - 2 * nu_b)
b_b = E_b * 1 / 2 / (1 + nu_b)

# initial and final porosity
phi_f_0 = 1 - 0.32
phi_f_inf = 1 - 0.64

# computational parameters
Nt = 100
dt = 1e-5 / 5

params = {"epsilon": epsilon, "delta": delta, "E": E,
            "E_b": E_b, "nu_d": nu_d, "nu_b": nu_b, "a_b": a_b, "a_d": a_d,
            "b_d": b_d, "b_b":b_b, "phi_f_0": phi_f_0, "phi_f_inf": phi_f_inf,
            "Q": Q, "Nt":Nt, "dt": dt}

beam_end = Point(1, 0)
t = np.zeros(Nt+1)
w = np.zeros(Nt+1)
Vol = np.ones(Nt+1)

disp_name = 'disp_volume.csv'

snes_solver_parameters = {"snes_solver": {"linear_solver": "mumps",
                                          "maximum_iterations": 8,
                                          "report": True,
                                          "error_on_nonconvergence": True}}

parameters["ghost_mode"] = "shared_facet"


output = XDMFFile("output/" + fname + "drop_beam.xdmf")
output.parameters["rewrite_function_mesh"] = False
output.parameters["functions_share_mesh"] = True
output.parameters["flush_output"] = True




p = Path('output/' + fname)
p.mkdir(parents = True, exist_ok = True)
p_file = open('output/' + fname + 'params.json', "w")
json = json.dump(params, p_file)
p_file.close()


#---------------------------------------------------------------------
# problem geometry: mesh and boundaries
#---------------------------------------------------------------------
mesh = Mesh('meshes/'+meshname+'.xml')
subdomains = MeshFunction("size_t", mesh, 'meshes/' + meshname + '_physical_region.xml')
bdry = MeshFunction("size_t", mesh, 'meshes/' + meshname + "_facet_region.xml")

# define the boundaries (values from the gmsh file)
free_surface = 1
substrate = 2
wall = 3

# define the domains
drop = 4
beam = 5

Od = generate_subdomain_restriction(mesh, subdomains, drop)
Ob = generate_subdomain_restriction(mesh, subdomains, beam)
Sig = generate_interface_restriction(mesh, subdomains, {drop, beam})

dx = Measure("dx", domain=mesh, subdomain_data=subdomains)
ds = Measure("ds", domain=mesh, subdomain_data=bdry)
dS = Measure("dS", domain=mesh, subdomain_data=bdry)
dS = dS(substrate)

# normal and tangent vectors
nn = FacetNormal(mesh); tt = as_vector((-nn[1], nn[0]))

#---------------------------------------------------------------------
# elements, function spaces, and test/trial functions
#---------------------------------------------------------------------
element_u = VectorElement("CG", mesh.ufl_cell(), 2)
element_p = FiniteElement("CG", mesh.ufl_cell(), 1)
element_l = VectorElement("DGT", mesh.ufl_cell(), 1)
mixed_element = BlockElement(element_u, element_p, element_p, element_u, element_l)
V = BlockFunctionSpace(mesh, mixed_element, restrict = [Od, Od, Od, Ob, Sig])

# unknowns and test functions
Y = BlockTestFunction(V)
(v_d, q, CC, v_b, eta) = block_split(Y)

Xt = BlockTrialFunction(V)

X = BlockFunction(V)
(u_d, p, C, u_b, lam) = block_split(X)

X_old = BlockFunction(V)
(u_d_old, p_old, C_old, u_b_old, lam_old) = block_split(X_old)

#---------------------------------------------------------------------
# boundary conditions
#---------------------------------------------------------------------
bc_wall = DirichletBC(V.sub(3), (0,0), bdry, wall)

bcs = BlockDirichletBC([bc_wall])


#---------------------------------------------------------------------
# Define kinematics
#---------------------------------------------------------------------
I = Identity(2)

# Drop
F_d = I + grad(u_d)
H_d = inv(F_d.T)
J_d = det(F_d)

# Beam
F_b = I + grad(u_b)
E = 0.5 * (F_b.T * F_b - I)

#---------------------------------------------------------------------
# define transport quantities
#---------------------------------------------------------------------

# Eulerian fluid fraction (porosity)
phi_f = C / J_d

# Permeability tensor
K = J_d * inv(F_d.T * F_d)

# Evaporation rate
q_evap = Q * (phi_f - phi_f_inf)

# Transformation of surface element
s_fact = J_d * sqrt(dot(H_d * nn, H_d * nn))

#---------------------------------------------------------------------
# define PK1 stress tensors
#---------------------------------------------------------------------

# drop
S_d = a_d * J_d * (J_d - 1) * H_d + b_d * (F_d - H_d) - p * J_d * H_d

# beam
S_b = F_b * (a_b * tr(E) * I + 2 * b_b * E)


#---------------------------------------------------------------------
# build equations
#---------------------------------------------------------------------

# Stress balance in drop
FUN1 = -inner(S_d, grad(v_d)) * dx(drop) + inner(lam("+"), v_d("+")) * dS

# Incompressibility for drop
ic = J_d - (1 + C - phi_f_0)
FUN2 = ic * q * dx(drop)

# Fluid transport in drop
FUN3 = (C - C_old) / dt * CC * dx(drop) + inner(K * grad(p), grad(CC)) * dx(drop) + q_evap * s_fact * CC * ds(free_surface)

# Stress balance in beam
FUN4 = -inner(S_b, grad(v_b)) * dx(beam) - inner(lam("-"), v_b("-")) * dS

# Continuity of beam/drop displacement
FUN5 = inner(avg(eta), (u_d("+") - u_b("-"))) * dS


# Total residual and Jacobian
FUN = [FUN1, FUN2, FUN3, FUN4, FUN5]
JAC = block_derivative(FUN, X, Xt)


#---------------------------------------------------------------------
# set up the solver
#---------------------------------------------------------------------

# Initialize solver
problem = BlockNonlinearProblem(FUN, X, bcs, JAC)
solver = BlockPETScSNESSolver(problem)
solver.parameters.update(snes_solver_parameters["snes_solver"])

# extract solution components
(u_d, p, C, u_b, lam) = X.block_split()

x = np.linspace(0, 1, 50)


# Initial conditions
C_old.interpolate(Constant(phi_f_0))
C.interpolate(Constant(phi_f_0))

# Name the variables (so these are imported into Paraview)
u_d.rename("u_d", "u_d")
u_b.rename("u_b", "u_b")
p.rename("p", "p")
C.rename("C", "C")

# Save initial state
output.write(u_d, 0)
output.write(u_b, 0)
output.write(p, 0)
output.write(C, 0)

# Compute initial drop volume
Vol[0] = assemble(J_d * dx(drop))


# Stuff for calculating and saving the stresses
DG1 = FunctionSpace(mesh, 'DG', 1)
Vs = BlockFunctionSpace([DG1], restrict = [Od])

Sxx = project(S_d[0,0], Vs.sub(0))
Sxx.rename("Sxx", "Sxx")
output.write(Sxx, 0)


#---------------------------------------------------------------------
# Time stepping
#---------------------------------------------------------------------
for i in range(Nt):

    print('-------------------------------------------')
    print('iteration', i+1, 'of', Nt)

    (its, conv) = solver.solve()

    if conv:

        # Compute stresses
        Sxx = project(S_d[0,0], Vs.sub(0))

        # update nominal fluid fraction
        C_old.assign(C)

        # Compute drop volume
        Vol[i+1] = assemble(J_d * dx(drop))

        # Compute deflection of beam end
        t[i+1] = t[i] + dt
        w[i+1] = u_d(beam_end)[1]
        np.savetxt('output/' + fname + disp_name, np.array([t[0:i+2], w[0:i+2], Vol[0:i+2]]), delimiter=",")

        # Name solution components and save
        u_d.rename("u_d", "u_d")
        u_b.rename("u_b", "u_b")
        p.rename("p", "p")
        C.rename("C", "C")
        Sxx.rename("Sxx", "Sxx")

        output.write(u_d, t[i+1])
        output.write(u_b, t[i+1])
        output.write(p, t[i+1])
        output.write(C, t[i+1])
        output.write(Sxx, t[i+1])

        # tau_x = np.array([lam((xx, 0))[0] for xx in x ])
        # tau_z = np.array([lam((xx, 0))[1] for xx in x ])
        # np.savetxt('output/' + fname + 'tau_' + str(i) + '.csv', np.array([x, tau_x, tau_z]), delimiter = ',')

        # Compute the nominal fluid fraction on the substrate and save
        C_bottom = np.array([C((xx, 0)) for xx in x ])
        np.savetxt('output/' + fname + 'C_bottom_' + str(i) + '.csv', np.array([x, C_bottom]), delimiter = ',')


        i += 1

    else:
        print('NO CONVERGENCE')
