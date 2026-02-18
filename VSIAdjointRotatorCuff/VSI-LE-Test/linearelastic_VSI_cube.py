# from mechanoChemML.src.graph_utilities import mesh
from ufl import *
from dolfin import *
import numpy as np
import random
import sys
from functools import reduce
# from stepwise_minimization_HGOConverged import *
from numpy import linalg as LA
from boundaryConditions import Cube5mm
from itertools import product
import os
from scipy.optimize import minimize

# number of constitive parameters/terms tracked
num_theta=2
# number of basis terms used in the residual weak residual
num_basis=2
# creates a global list of 12 FEniCS Constants theta to hold the constitutive parameters.
# These are parameters the weak form will depend on.
theta=[]
for i in range(num_theta):
  theta.append(Constant(0.0))


def nonlinear_VSI_LE_cube(target_array, mesh_xdmf, inp_path, u_csv_path, rf2_top,
                          loss_factor2, activate_basis_index, bc_mode):
  #################################### Define mesh ############################
  """
  nonlinear_VSI_HGO is a function that performs a nonlinear optimization of a VSI constitutive model using the HGO optimization algorithm.

  The function takes in the following arguments:
    target_array: an array of the current target values for the constitutive parameters
    mesh_xdmf: the path to the XDMF mesh file
    u_csv: the path to the CSV file containing displacement data
    rf2_top: the top boundary region for traction forces
    loss_factor2: the weight of the loss term for the external force
    meshName: the name of the mesh
    current_activate_index: an array of the indices of the currently active constitutive parameters
    target_index: the index of the currently active constitutive parameter
  
  Evaluate the scalar loss for a given set of constitutive parameters and a
  single tendon loading case within a variational system identification (VSI)
  framework.

  This function constructs a finite-deformation hyperelastic weak form using
  a library of candidate strain energy density terms (including volumetric,
  isochoric, and anisotropic HGO-type contributions). Boundary regions are
  identified using geometry-based facet classification rules defined in
  `boundaryConditions.py`, and measured displacement fields are imposed as
  Dirichlet boundary conditions.

  For the specified tendon, condition, and displacement level, the function:
    1) assigns the active constitutive parameters to the weak form,
    2) assembles the residual of the equilibrium equations,
    3) computes the traction on the tendon boundary,
    4) evaluates a loss composed of (i) the normalized residual norm and
        (ii) the normalized mismatch between predicted and measured external
        forces.

  The function performs no optimization itself and returns only the scalar
  loss value. It is intended to be called repeatedly by an external optimizer
  or stepwise regression routine during constitutive model identification.
  """
  
  debug = False
  # Creates empty mesh object
  mesh=Mesh()
  # Reads XDMF mesh from path
  with XDMFFile(mesh_xdmf) as infile:
    infile.read(mesh)

  V = VectorFunctionSpace(mesh, "Lagrange", 1)  # CG1 for nodal ABAQUS data
  
  # FE Space for vector fields (displacements) using CG2 (quadratic Lagrange)
  # V = VectorFunctionSpace(mesh, "Lagrange", 2)

  # Return symbolic coordiante tuple for use in forms
  x = SpatialCoordinate(mesh)
  dof_coordinates = V.tabulate_dof_coordinates()                    
  dof_coordinates.resize((V.dim(), mesh.geometry().dim()))                          
  
  ######################### Define regions to apply boundary conditions  ##########################
  dim = mesh.topology().dim()
  facetfct = MeshFunction('size_t', mesh, dim - 1)
  facetfct.set_all(0)

  # Initialize the topological connectivity between facets and cells
  mesh.init(dim - 1, dim) # relates facets to cells

  # Dictionary of boundary conditions for tendon

  equationDict = Cube5mm["equations"][bc_mode].items()
  keyList = list(Cube5mm["equations"][bc_mode].keys())

  # iterates through facets and applies boundary conditions
  for f in facets(mesh):
    for key, equation in equationDict:
      if equation(f):
        facetfct[f.index()] = keyList.index(key) + 1

  #Gives symbolic outward normal vector on the boundary facets
  n = FacetNormal(mesh)
  # Set quadrature degree for integration
  q_degree = 5

  ########################### Define functions, invariants, fiber direction ######################################
  # Creates a fields tha can hold FE coefficients (displacements)
  # u = Function(V)
  # with XDMFFile(mesh_xdmf) as infile:
  #   infile.read_checkpoint(u, "U", 0) 

  # Load u from Abaqus CSV using INP node coordinates
  u = load_u_into_function_from_inp_csv(mesh, V, inp_path, u_csv_path, tol=1e-6)

  #Creates a test function for variational forms
  v = TestFunction(V)

  for i in range(len(theta)):
    theta[i].assign(0.0)
   
   # Assign the active parameters from target_array to theta[i] slots
   # So target_array is a compressed parameter vector, anc current_activate_index tells
   # us where to put these values in the full theta list
  target_array_index=0
  k = 0

  for k,i in enumerate(activate_basis_index):
    theta[i].assign(target_array[k])
    # k += 1
  
  # Kinematics
  # d is the spatial dimension
  # d=len(u)
  d = mesh.geometry().dim()
  I = Identity(d)          # Identity tensor

  def eps(w):
      return sym(grad(w))
  
  lam = theta[0]  # lamda
  mu = theta[1]   # mu

  sigma = lam*tr(eps(u))*I + 2*mu*eps(u) # linear elastic stress for testing

  # Bounndary integration measure on the "tendon" boundary, integrate traction and compute predicted force
  ds_top = Measure("ds", domain=mesh, subdomain_data=facetfct,
                 subdomain_id=keyList.index("TopFace")+1,
                 metadata={'quadrature_degree': q_degree})

  ################## Candidate functions for strain energy density ###########################

  basis_pool = [0]*2
  basis_pool[0] = theta[0] * tr(eps(u))*tr(eps(v)) * dx(metadata={'quadrature_degree': q_degree})
  basis_pool[1] = theta[1] * 2.0*inner(eps(u), eps(v)) * dx(metadata={'quadrature_degree': q_degree})
  R = basis_pool[0] + basis_pool[1]


  # Define residual
  # Total weak residual. If the constitutive model is correct, the residual should be close to zero.
  # VSI minimizes the residual norm
  R=0
  for term in basis_pool:
    R += term

  ########################### Define displacement and force for each tendon ##################

  ##################################### VSI implementation ###################################
  
  # Resets all coefficients to zero, then sets the "target" coefficient to 1.0
  # for i in range(len(theta)):
  #   theta[i].assign(0.0)
  # theta[target_index].assign(1.0)
  
  n= FacetNormal(mesh)
  traction = dot(sigma, n)
  load_dir = Constant((0.0, 1.0, 0.0))
  F_pred = assemble(dot(traction, load_dir)*ds_top)
  # print("Predicted force = ", F_pred)
  # print("theta =", float(theta[0]), float(theta[1]), "F_pred =", float(F_pred), "F_meas =", float(rf2_top))
  F_meas = rf2_top
  # print("Measured force = ", F_meas)

  A_top = assemble(1.0*ds_top)
  # print("A_top =", A_top)


  # Assign displacement 
  normal_to_surf = n
  
  # Calculate traction, external force, and residual

  tem=assemble(R)

  # Apply boundary conditions

  # for key in ["BottomFace", "TopFace"]:
  #   bc = DirichletBC(V, u, facetfct, keyList.index(key) + 1)
  #   bc.apply(tem)

  bc_keys = ["BottomFace", "TopFace"]
  for extra in ["XMin", "XMax", "ZMin", "ZMax"]:
      if extra in keyList:
          bc_keys.append(extra)

  for key in bc_keys:
      bc = DirichletBC(V, u, facetfct, keyList.index(key) + 1)
      bc.apply(tem)

  
  tem=tem[:]

  # Weak Form Residual
  # Measures how well the displacement field satisfies the equilibrium equations for the current material parameters.
  loss1 = np.inner(tem,tem)/(np.size(tem)*F_meas**2)

  # Force-matching penalty term, compares computed boundary resultant to measured force
  # Ensures the boundary traction predicted by the model matches the measured reaction force.
  loss2 = (F_pred - F_meas)**2/(F_meas**2)

  loss_factor1 = 1.
  # print("loss1 = ", loss_factor1*loss1)
  # print("loss2 = ", loss_factor2*loss2)

  if debug and MPI.rank(MPI.comm_world) == 0:
    print("theta =", float(theta[0]), float(theta[1]))
    print("F_pred =", float(F_pred))
    print("F_meas =", float(F_meas))
    print("A_top =", float(A_top))
    print("loss1 =", float(loss1))
    print("loss2 =", float(loss2))


  lam=0. # penalty term
  a = [0]*len(theta)
  penaltySum = 0
  for i in range(len(theta)):
    a[i]= theta[i].values()
    penaltySum += a[i]**2

  # Total loss combining physical residual, force-matching penalty, and regularization
  loss = loss_factor1*loss1 + loss_factor2*loss2 + 0.5*lam*penaltySum
  return loss 

# def add_up_losses(target_array, tendonStamp, meshName, loss_factor2, combination, current_activate_index,
#                       target_index):
#   """
#   Compute the total objective (loss) across multiple loading cases for a single
#   tendon by summing the per-case losses returned by `nonlinear_VSI_HGO`.

#   This function is used as the scalar objective callable passed to the external
#   optimizer/stepwise elimination routine. For each (condition, displacement)
#   pair in `combination`, it evaluates `nonlinear_VSI_HGO(...)` using the same
#   constitutive parameter vector and active-term definition, and accumulates the
#   resulting scalar losses.

#   Parameters
#   ----------
#   target_array : array_like, shape (n_active,)
#       Current values of the constitutive coefficients for the *active* terms
#       only. These values are assigned into the global `theta` list at indices
#       given by `current_activate_index` inside `nonlinear_VSI_HGO`. The ordering
#       of `target_array` must match the ordering of `current_activate_index`.

#   tendonStamp : str
#       Tendon identifier string used to locate data on disk and to select the
#       appropriate boundary classification rules from `boundaryConditions.py`
#       via `globals()[tendonStamp]`. In this repository it is typically of the
#       form "TendonYYYYMMDD" (e.g., "Tendon20231012").

#   meshName : str
#       Mesh resolution/name token used to load the corresponding mesh and field
#       files (e.g., "Coarse"). This is used in file paths inside
#       `nonlinear_VSI_HGO`.

#   loss_factor2 : float
#       Weight applied to the external force/traction mismatch term (loss2) in
#       the per-case loss computed by `nonlinear_VSI_HGO`. Larger values place
#       more emphasis on matching measured external force.

#   combination : array_like of shape (n_cases, 2)
#       List/array of loading cases to evaluate and sum over. Each entry should
#       be a pair (condition, disp), where:
#         - condition is a string such as "Intact" or "Torn"
#         - disp is a displacement-level label (often convertible to str)
#       The function loops over these pairs and sums the corresponding per-case
#       losses.

#   current_activate_index : array_like of int, shape (n_active,)
#       Indices into the full candidate parameter vector `theta` that define
#       which constitutive terms are currently active (included) in the model.
#       These indices determine where entries of `target_array` are assigned.

#   target_index : int
#       Index of the "target" (reference) term in the full `theta` vector that
#       is held fixed (set to 1.0) inside `nonlinear_VSI_HGO` to normalize the
#       model scaling.

#   Returns
#   -------
#   losses : float
#       Scalar total loss equal to the sum of per-case losses across all entries
#       of `combination`.
#   """
#   losses = 0.0

#   for i in range(len(combination)):
#       losses += nonlinear_VSI_LE_cube(target_array, tendonStamp, str(combination[i][0]), 
#                                   str(combination[i][1]), loss_factor2, meshName, current_activate_index,
#                                   target_index)
#   return losses
  
import numpy as np

def objective_two_cases(target_array, cases, mesh_xdmf, loss_factor2, activate_basis_index):
    total = 0.0
    for case in cases:
        total += nonlinear_VSI_LE_cube(
            target_array,
            mesh_xdmf,
            case["inp_path"],
            case["u_csv"],
            case["rf2_top"],
            loss_factor2,
            activate_basis_index,
            case["bc_mode"],
        )
    return total

def read_inp_nodes(inp_path):
    """Return dict: nodeLabel -> (x,y,z) from Abaqus .inp *NODE section."""
    node_coords = {}
    in_nodes = False
    with open(inp_path, "r", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.upper().startswith("*NODE"):
                in_nodes = True
                continue
            if in_nodes and s.startswith("*"):
                break
            if in_nodes:
                parts = [p.strip() for p in s.split(",")]
                if len(parts) >= 4:
                    try:
                        lab = int(parts[0])
                        node_coords[lab] = (float(parts[1]), float(parts[2]), float(parts[3]))
                    except:
                        pass
    if not node_coords:
        raise RuntimeError("Could not parse *NODE section from {}".format(inp_path))
    return node_coords

def read_cube_u_csv(u_csv_path):
    """Return dict: nodeLabel -> (U1,U2,U3) from cube_U.csv."""
    data = np.genfromtxt(u_csv_path, delimiter=",", names=True)
    # handle single-row
    if data.shape == ():
        data = np.array([data])
    u_by_label = {}
    for row in data:
        lab = int(row["nodeLabel"])
        u_by_label[lab] = (float(row["U1"]), float(row["U2"]), float(row["U3"]))
    return u_by_label

def key_xyz(xyz, tol=1e-9):
    """Hashable coordinate key with tolerance."""
    return (round(xyz[0]/tol)*tol, round(xyz[1]/tol)*tol, round(xyz[2]/tol)*tol)

def load_u_into_function_from_inp_csv(mesh, V, inp_path, u_csv_path, tol=1e-9):
    """
    Build Function u in Vector CG1 space by:
      INP: nodeLabel -> xyz
      CSV: nodeLabel -> U
      Map xyz -> U
      Assign into u.vector() using subspace dofmaps.
    """
    node_xyz = read_inp_nodes(inp_path)
    u_by_label = read_cube_u_csv(u_csv_path)

    # coord -> displacement vector
    xyz_to_u = {}
    for lab, xyz in node_xyz.items():
        if lab in u_by_label:
            xyz_to_u[key_xyz(xyz, tol)] = u_by_label[lab]

    u = Function(V)
    values = u.vector().get_local()

    dof_coords = V.tabulate_dof_coordinates().reshape((-1, mesh.geometry().dim()))

    # DOFs for each component
    dofs0 = V.sub(0).dofmap().dofs()
    dofs1 = V.sub(1).dofmap().dofs()
    dofs2 = V.sub(2).dofmap().dofs()

    def assign_component(dofs, comp):
        missing = 0
        for dof in dofs:
            xyz = dof_coords[dof]
            k = key_xyz(xyz, tol)
            if k not in xyz_to_u:
                missing += 1
                continue
            values[dof] = xyz_to_u[k][comp]
        return missing

    m0 = assign_component(dofs0, 0)
    m1 = assign_component(dofs1, 1)
    m2 = assign_component(dofs2, 2)

    if (m0 + m1 + m2) > 0:
        raise RuntimeError(
            "Failed to map {} DOFs to node displacements. Try tol=1e-6 or check coordinate units."
            .format(m0 + m1 + m2)
        )

    u.vector().set_local(values)
    u.vector().apply("insert")
    return u

def read_rf2_top(path):
        with open(path, "r") as f:
            for line in f:
                if "RF2_top" in line:
                    # Handles both "RF2_top = -25" and "RF2_top,-25"
                    parts = line.replace("=", ",").split(",")
                    return float(parts[-1])
        raise RuntimeError("RF2_top not found in file")

###################### Here's where the code execution starts ####################
"""
IN THIS CASE:
coeffs0[0] = bulk modulus, volumetric
coeffs0[1] = isochoric I1, stiffness
coeffs0[2] = anisotropic exp prefactor
coeffs0[3] = anisotropic exp sharpness
coeffs0[4] = fiber dispersion K [0, 1/3] is a classical fiber-dispersion parameter constraint
coeffs0[5] = (I1-3)^2
coeffs0[6] = (I1-3)^3
coeffs0[7] = (I1-3)^4
coeffs0[8] = (I2-3)^2
coeffs0[9] = (I2-3)^3
coeffs0[10] = (I2-3)^4

IN THIS CASE:
theta 0 - 2, core physics (never eliminated)
theta 3 - 4, anisotropic shape control
theta 5 - 10, higher-order isotropic terms (subject to elimination)
"""
if __name__ == "__main__":

  print("===== Linear Elastic VSI on ABAQUS Cube =====")

  # designate index 11 as "target" term
  # target_index=11
  num_theta = 2
  activate_basis_index=[0,1]
  coeffs0 = np.array([1., 1.])
  mesh_xdmf = "cube_with_U.xdmf"
  rf2_case1 = read_rf2_top("cube_RF2_case1.txt")
  rf2_case2 = read_rf2_top("cube_RF2_case2.txt")

  loss_factor2 = 1e-3
  cases = [
    {
        "name": "case1",
        "bc_mode": "Case1_Unconfined",   # must match your updated boundaryConditions.py
        "inp_path": "TestVSI-Tets.inp",
        "u_csv": "cube_U_case1.csv",
        "rf2_top": rf2_case1,
    },
    {
        "name": "case2",
        "bc_mode": "Case2_Confined",     # must match your updated boundaryConditions.py
        "inp_path": "TestVSI-Tets2.inp",
        "u_csv": "cube_U_case2.csv",
        "rf2_top": rf2_case2,
    },
]
  
  # Creates bounds, default lower bound 0, upper bound infinity
  bounds = np.zeros((coeffs0.size,2))
  bounds[:,0] = 0.#1e-5	#All terms positive
  bounds[:,1] = np.inf
  args = (cases, mesh_xdmf, loss_factor2, activate_basis_index)

  print("loss at initial guess (two cases):",
      objective_two_cases(coeffs0, *args))
  
  # print("loss at initial guess:", nonlinear_VSI_LE_cube(coeffs0, mesh_xdmf, rf2_top, loss_factor2, activate_basis_index))
  # print("F_pred at [1,1] =", nonlinear_VSI_LE_cube(np.array([1.0,1.0]), mesh_xdmf, rf2_top, loss_factor2, activate_basis_index))
  # print("F_pred at [2,1] =", nonlinear_VSI_LE_cube(np.array([2.0,1.0]), mesh_xdmf, rf2_top, loss_factor2, activate_basis_index))
  # print("F_pred at [1,2] =", nonlinear_VSI_LE_cube(np.array([1.0,2.0]), mesh_xdmf, rf2_top, loss_factor2, activate_basis_index))

  
  # folder_name = tendonStampList[index] + '/results/HGOHighOrderI1I2_SquaredNorm'

  # if not os.path.exists(folder_name):
  #     os.makedirs(folder_name)
  
  # save_to_file= folder_name + '/NoStag_' + identifier + '_CG2.dat'

  # print(save_to_file)
  method_options={'disp':True,
                  'ftol':1.0e-12,
                  'eps': 1e-8,
                  'maxiter': 1000 }

  print("Running optimization...")
  
  result = minimize(
    objective_two_cases,
    coeffs0,
    args=args,
    method="SLSQP",
    bounds=bounds,
    options=method_options,
  )
  
  print("\n===== Optimization Complete =====")
  print("Optimized lambda =", result.x[0])
  print("Optimized mu     =", result.x[1])
  print("Final loss       =", result.fun)
 

    # Run stepwise minimization and saves the coefficient path and elimination history
    # gamma_matrix, loss=stepwise_minimization_HGOConverged(add_up_losses, coeffs0, 
    #                                             args_dict=args, grad_f=[],
    #                                             method_options=method_options,
    #                                             save_to_file=save_to_file)
    
    # save_gamma= folder_name + '/gamma_NoStag_' + identifier + '_CG2.dat'
    # np.savetxt(save_gamma,gamma_matrix)