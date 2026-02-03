from ufl import *
from dolfin import *
import numpy as np
import random
import sys
from functools import reduce
from stepwise_minimization_HGOConverged import *
from numpy import linalg as LA
from boundaryConditions import *
from itertools import product
import os

# number of constitive parameters/terms tracked
num_theta=12
# number of basis terms used in the residual weak residual
num_basis=9
# creates a global list of 12 FEniCS Constants theta to hold the constitutive parameters.
# These are parameters the weak form will depend on.
theta=[]
for i in range(num_theta):
  theta.append(Constant(0.0))


def nonlinear_VSI_HGO(target_array, tendonName, condition, disp, loss_factor2, meshName, current_activate_index,
                      target_index):
  #################################### Define mesh ############################
  """
  nonlinear_VSI_HGO is a function that performs a nonlinear optimization of a VSI constitutive model using the HGO optimization algorithm.

  The function takes in the following arguments:
    target_array: an array of the current target values for the constitutive parameters
    tendonName: the name of the tendon
    condition: the condition of the experiment, e.g. 'Intact' or 'Torn'
    disp: the displacement case of the tendon
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

  tendonDate = tendonName[6:]
  
  # Creates empty mesh object
  mesh=Mesh()
  # Reads XDMF mesh from path
  with XDMFFile(tendonDate + "/convergenceAnalysis_V2/mesh/Tendon" + tendonDate+ 
                "_" + condition + "_" + meshName + ".xdmf") as infile:
    infile.read(mesh)
  
  # FE Space for vector fields (displacements) using CG2 (quadratic Lagrange)
  V = VectorFunctionSpace(mesh, "Lagrange", 2)

  # Return symbolic coordiante tuple for use in forms
  x=SpatialCoordinate(mesh)
  dof_coordinates = V.tabulate_dof_coordinates()                    
  dof_coordinates.resize((V.dim(), mesh.geometry().dim()))                          
  
  ######################### Define regions to apply boundary conditions  ##########################
  dim = mesh.topology().dim()
  # print("dim = ", V.dim())
  # field of labels attached to mesh entities. It is a lookup table
  # dim-1 means we are defining the facets of the mesh. In a 3D mesh, facets are 2D surfaces.
  facetfct = MeshFunction('size_t', mesh, dim - 1)
  # Initialize all facets to 0, then nonzero values will be assigned based on
  # boundary conditions
  facetfct.set_all(0)
  # Initialize the topological connectivity between facets and cells
  mesh.init(dim - 1, dim) # relates facets to cells

  # Dictionary of boundary conditions for tendon
  equationDict = globals()[tendonName]["equations"][condition].items()
  keyList = list(globals()[tendonName]["equations"][condition].keys())
  
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
  u = Function(V,name='u') 

  #Creates a test function for variational forms
  v = TestFunction(V)
  
  # Kinematics
  # d is the spatial dimension
  d=len(u)
  I = Identity(d)          # Identity tensor
  F = I + grad(u)          # Deformation gradient    
  
  J=det(F)
  C = F.T*F               # Right Cauchy-Green tensor
  B = F*F.T               # Left Cauchy-Green tensor

  # Invariants
  invC = inv(C)              
  I1 = tr(C)                  # volumetric
  I2 = 0.5*(I1*I1-tr(C*C) )     # biaxial state
  I3 = det(C)                # triaxial

  # Fiber direction
  file_UVW = HDF5File(MPI.comm_world, tendonDate + 
                      "/convergenceAnalysis_V2/UVWHighRes/Tendon" + tendonDate +
                      "_" + meshName + "_UVW_" + condition + "_CG2.h5", 'r')

  a = Function(V, name='UVW')
  file_UVW.read(a, '/UVW_' + condition)

  # Computes fiber stretch invariant I4, a is the local fiber direction vector field
  I4=dot(a,C*a)

  # Isochoric invariants (volume preserving), logJ logarithmic volume change
  barI1=J**(-2./3.)*I1
  barI2=J**(-4./3.)*I2
  logJ = ln(J)
  
  # Bounndary integration measure on the "tendon" boundary, integrate traction and compute predicted force
  dss = Measure("ds", domain=mesh, subdomain_data=facetfct,
                subdomain_id=keyList.index("tendon")+1, metadata={'quadrature_degree': q_degree})

  # Mixed isotropic-anisotropic invariant. Weighted blend of isotropic nad fiber stretch
  I14 = theta[4]*I1+(1.-3.*theta[4])*I4-1.

  ################## Candidate functions for strain energy density ###########################
  # Volumetric stress contribution
  hS0 = (J-1)*J*invC
  # hS0 = 0.5*logJ*invC

  # Isochoric Neo-Hookean stress
  hS1 = (J**(-2./3.)*I-1./3.*barI1*invC)
  # hS1 = 0.5*I - 0.5*invC - (1./3.)*logJ*invC

  # Compressible anisotropic part
  # Anisotropic HGO fiber stress
  hS2 = theta[3]*I14*exp(theta[3]*I14*I14)*(theta[4]*I + (1.-3.*theta[4])*outer(a,a))

  # I1 terms
  # Higher-order isotropic terms based on I1. Allows VSI to discover wheather nonlinear 
  # isotropic terms are needed for I1.
  hS3 = 2*(barI1-3)*(J**(-2./3.)*I-1./3.*barI1*invC) # quadratic
  hS4 = 3*(barI1-3)**2*(J**(-2./3.)*I-1./3.*barI1*invC) # cubic
  hS5 = 4*(barI1-3)**3*(J**(-2./3.)*I-1./3.*barI1*invC) # fourth order

  # I2 terms
  # Higher-order isotropic terms based on I2. Allows VSI to discover wheather nonlinear
  # isotropic terms are needed but for I2.
  hS6 = 2*(barI2-3)*(J**(-2./3.)*barI1*I- J**(-4./3.)*C-2./3.*barI2*invC) # quadratic
  hS7 = 3*(barI2-3)**2*(J**(-2./3.)*barI1*I- J**(-4./3.)*C-2./3.*barI2*invC) # cubic
  hS8 = 4*(barI2-3)**3*(J**(-2./3.)*barI1*I- J**(-4./3.)*C-2./3.*barI2*invC) # fourth order

  # Here we build the basis function
  # Library of candidate weak-form terms. Each term represents a possible physical mechanism. VSI
  # will turn terms on or off via theta[i] coefficients.
  basis_pool=[0]*num_basis
  basis_pool[0]=theta[0]*inner(2*F*hS0, grad(v) )*dx(metadata=
                                                     {'quadrature_degree': q_degree})
  basis_pool[1]=theta[1]*inner(2*F*hS1, grad(v) )*dx(metadata=
                                                     {'quadrature_degree': q_degree})
  basis_pool[2]=theta[2]*inner(2*F*hS2, grad(v) )*dx(metadata=
                                                     {'quadrature_degree': q_degree})
  basis_pool[3]=theta[5]*inner(2*F*hS3, grad(v) )*dx(metadata=
                                                     {'quadrature_degree': q_degree})
  basis_pool[4]=theta[6]*inner(2*F*hS4, grad(v) )*dx(metadata=
                                                     {'quadrature_degree': q_degree})
  basis_pool[5]=theta[7]*inner(2*F*hS5, grad(v) )*dx(metadata=
                                                     {'quadrature_degree': q_degree})
  basis_pool[6]=theta[8]*inner(2*F*hS6, grad(v) )*dx(metadata=
                                                     {'quadrature_degree': q_degree})
  basis_pool[7]=theta[9]*inner(2*F*hS7, grad(v) )*dx(metadata=
                                                     {'quadrature_degree': q_degree})
  basis_pool[8]=theta[10]*inner(2*F*hS8, grad(v) )*dx(metadata=
                                                     {'quadrature_degree': q_degree})

  # Define residual
  # Total weak residual. If the constitutive model is correct, the residual should be close to zero.
  # VSI minimizes the residual norm
  R=0
  for i in range(len(basis_pool)):
    R+=basis_pool[i]

  ########################### Define displacement and force for each tendon ##################
  # loads displacement field from HDF5 file
  file_U = HDF5File(MPI.comm_world, tendonDate + 
                    "/convergenceAnalysis_V2/displacementHighRes/Tendon" + tendonDate +
                    "_" + meshName + "_U_" + condition + "_" + str(disp) + "_CG2.h5", 'r')
  # loads force measurement from text file
  force_used = np.loadtxt(tendonDate + "/forceData/" + tendonDate + 
                          "_MedianForceData_" + condition + "_" + str(disp) + "mm.txt")

  # First-Piola-Kirchhoff stress tensor
  P=2*F*(theta[0]*hS0 + theta[1]*hS1 + theta[2]*hS2 + theta[5]*hS3 + theta[6]*hS4 
         + theta[7]*hS5 + theta[8]*hS6 + theta[9]*hS7 + theta[10]*hS8)

  # Holds boundary displacement values
  boneDisp = Function(V, name='boneDisp')

  ##################################### VSI implementation ###################################
  
  # Resets all coefficients to zero, then sets the "target" coefficient to 1.0
  for i in range(len(theta)):
    theta[i].assign(0.0)
  theta[target_index].assign(1.0)
   
   # Assign the active parameters from target_array to theta[i] slots
   # So target_array is a compressed parameter vector, anc current_activate_index tells
   # us where to put these values in the full theta list
  target_array_index=0
  for i in current_activate_index:
    theta[i].assign(target_array[target_array_index])
    target_array_index+=1
  
  # Assign displacement 
  # Loads displacement field into u
  file_U.read(u, 'U_' + condition + str(disp))
  file_U.read(boneDisp, 'U_' + condition + str(disp))
  normal_to_surf = n
  
  # Calculate traction, external force, and residual
  # Select load direction: here we apply uniaxial tension in -x direction
  load_dir = Constant((-1.0, 0.0, 0.0))
  traction = dot(P, normal_to_surf)
  # Calculate load on the "tendon" boundary
  loadOnFace = dot(traction, load_dir)*dss
  P_ext = force_used
  # Assemble the discrete weak form residual vector
  tem=assemble(R-dot(traction,v)*dss) 

  # Apply boundary conditions

  for key in keyList:
    bc_key = DirichletBC(V, boneDisp, facetfct, keyList.index(key) + 1)
    bc_key.apply(tem)
  
  tem=tem[:]
  # Physical residual loss, squared Euclidean nrom of assembled residual vector
  # Normalized by number of entries and by measured external force squared
  loss1 = np.inner(tem,tem)/(np.size(tem)*P_ext**2)
  # Force-matching penalty term, compares computed boundary resultant to measured force
  loss2 = pow(assemble(loadOnFace) - P_ext,2)/(P_ext**2) 
  loss_factor1 = 1.
#   print("loss1 = ", loss_factor1*loss1)
#   print("loss2 = ", loss_factor2*loss2)
  lam=0. # penalty term
  a = [0]*len(theta)
  penaltySum = 0
  for i in range(len(theta)):
    a[i]= theta[i].values()
    penaltySum += a[i]**2

  # Total loss combining physical residual, force-matching penalty, and regularization
  loss = loss_factor1*loss1 + loss_factor2*loss2 + 0.5*lam*penaltySum
  return loss 

def add_up_losses(target_array, tendonStamp, meshName, loss_factor2, combination, current_activate_index,
                      target_index):
  """
  Compute the total objective (loss) across multiple loading cases for a single
  tendon by summing the per-case losses returned by `nonlinear_VSI_HGO`.

  This function is used as the scalar objective callable passed to the external
  optimizer/stepwise elimination routine. For each (condition, displacement)
  pair in `combination`, it evaluates `nonlinear_VSI_HGO(...)` using the same
  constitutive parameter vector and active-term definition, and accumulates the
  resulting scalar losses.

  Parameters
  ----------
  target_array : array_like, shape (n_active,)
      Current values of the constitutive coefficients for the *active* terms
      only. These values are assigned into the global `theta` list at indices
      given by `current_activate_index` inside `nonlinear_VSI_HGO`. The ordering
      of `target_array` must match the ordering of `current_activate_index`.

  tendonStamp : str
      Tendon identifier string used to locate data on disk and to select the
      appropriate boundary classification rules from `boundaryConditions.py`
      via `globals()[tendonStamp]`. In this repository it is typically of the
      form "TendonYYYYMMDD" (e.g., "Tendon20231012").

  meshName : str
      Mesh resolution/name token used to load the corresponding mesh and field
      files (e.g., "Coarse"). This is used in file paths inside
      `nonlinear_VSI_HGO`.

  loss_factor2 : float
      Weight applied to the external force/traction mismatch term (loss2) in
      the per-case loss computed by `nonlinear_VSI_HGO`. Larger values place
      more emphasis on matching measured external force.

  combination : array_like of shape (n_cases, 2)
      List/array of loading cases to evaluate and sum over. Each entry should
      be a pair (condition, disp), where:
        - condition is a string such as "Intact" or "Torn"
        - disp is a displacement-level label (often convertible to str)
      The function loops over these pairs and sums the corresponding per-case
      losses.

  current_activate_index : array_like of int, shape (n_active,)
      Indices into the full candidate parameter vector `theta` that define
      which constitutive terms are currently active (included) in the model.
      These indices determine where entries of `target_array` are assigned.

  target_index : int
      Index of the "target" (reference) term in the full `theta` vector that
      is held fixed (set to 1.0) inside `nonlinear_VSI_HGO` to normalize the
      model scaling.

  Returns
  -------
  losses : float
      Scalar total loss equal to the sum of per-case losses across all entries
      of `combination`.
  """
  losses = 0.0

  for i in range(len(combination)):
      losses += nonlinear_VSI_HGO(target_array, tendonStamp, str(combination[i][0]), 
                                  str(combination[i][1]), loss_factor2, meshName, current_activate_index,
                                  target_index)
  return losses
  

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
  # designate index 11 as "target" term
  target_index=11
  activate_basis_index=[0,1,2,3,4,5,6,7,8,9,10]
  coeffs0=np.zeros(len(activate_basis_index))
  coeffs0 = np.array([1., 1., 1., 1., 1./3., 1., 1., 1., 1., 1., 1.]) 
  print("target_index=",target_index)
  
  meshName = "Coarse"
  conditionList = ["Intact", "Torn"]
  dispList = ["1", "2"]
  combinationList = np.array(list(product(conditionList, dispList)))
  combinationList = combinationList[1:,]
  # force term is a light constraint, weak-form residual dominates
  loss_factor2 = 1e-6
  tendonStampList = ["20231012", "20231017", "20231107", "20231114", "20231201", 
                     "20231206", "20231212", "20240503", "20240517", "20241127_1",
                     "20241127_2", "20241219"]
  indexList = [0,1,2,3,4,5,6,7,8,9,10,11]
  indexList = [0]
  
  identifier = 'LF1E_6_NoBounds_CoarseMesh'
  for index in indexList:
    print("Tendon ", tendonStampList[index])
    loss0 = add_up_losses(coeffs0, "Tendon" + tendonStampList[index], meshName, loss_factor2, 
                          combinationList, activate_basis_index, target_index)
    print("Loss0 = ", loss0)
    
    # Creates bounds, default lower bound 0, upper bound infinity
    bounds=np.zeros((coeffs0.size,2))
    bounds[:,0]= 0.#1e-5	#All terms positive
    bounds[:,1] = np.inf
    bounds[0,0] = 0.1 #volumetric coefficient lower bound
    bounds[1,0] = 0.04 #deviatoric coefficient lower bound
    bounds[3,0] = 1.e-4 #exponential coefficient lower bound
    bounds[4,1] = 1./3. #fiber dispersion upper bound
    num_activate_index=len(activate_basis_index)
    gamma_matrix=np.zeros((num_activate_index,num_activate_index))

    # Arguments dictionary for stepwise minimization
    # Freezes indices 0-4 (core physics and anisotropic terms)
    args={'num_theta':num_theta,'activate_basis_index':activate_basis_index,
            'target_vector_index':target_index,
            'frozen_index':[0,1,2,3,4], 'max_eliminate_step':6,
            'method':'SLSQP','bounds':bounds, 
            'tendonStamp': "Tendon" + tendonStampList[index], 
            'meshName': meshName, 'combination': combinationList, 
            'lossFactor': loss_factor2}
    
    folder_name = tendonStampList[index] + '/results/HGOHighOrderI1I2_SquaredNorm'

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    save_to_file= folder_name + '/NoStag_' + identifier + '_CG2.dat'

    print(save_to_file)
    method_options={'disp':False,'ftol':1.0e-15,'eps': 1e-10,'maxiter': 1000 }

    # Run stepwise minimization and saves the coefficient path and elimination history
    gamma_matrix, loss=stepwise_minimization_HGOConverged(add_up_losses, coeffs0, 
                                                args_dict=args, grad_f=[],
                                                method_options=method_options,
                                                save_to_file=save_to_file)
    
    save_gamma= folder_name + '/gamma_NoStag_' + identifier + '_CG2.dat'
    np.savetxt(save_gamma,gamma_matrix)