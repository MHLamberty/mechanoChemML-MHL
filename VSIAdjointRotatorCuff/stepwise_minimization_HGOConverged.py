import numpy as np
import scipy.optimize as sp
import os
import sys

"""
The goal of the stepwise_minimization_HGOConverged function is to perform a 
stepwise minimization of an objective function using the SciPy optimization library.
SciPy uses the HGO algorithm to perform the optimization. The function takes in an 
initial guess for the objective function, a set of arguments, and various options 
for the optimization method. The function iteratively eliminates basis functions 
from the optimization problem based on a threshold value for the change in the 
objective function. The function returns the final optimized objective function 
value and the corresponding basis function indices.
"""

def iter_cb(m):
  print ("results = ", m)
  
def stepwise_minimization_HGOConverged(obj_f, x0, args_dict,F_threshold=1.0e16, 
                                       method_options={}, grad_f=None, save_to_file=None):
  ##
  """
  Docstring for stepwise_minimization_HGOConverged
  
  :param obj_f: Objective function (loss function) to be minimized
  :param x0: Initial guess for the current active basis function coefficients
  :param args_dict: Dictionary of additional arguments for the objective function
      active terms, target vector index, number of theta, mesh name, tendon stamp,  
      combination, loss factor
  :param F_threshold: Threshold for change in objective function to eliminate basis 
      functions
  :param method_options: Options for the optimization method for SciPy's minimize function
      (optional)
  :param grad_f: Gradient of the objective function (optional). If not provided,
      SciPy will approximate the gradient numerically. jac=grad_f in sp.minimize
  :param save_to_file: Path to save intermediate results (optional).
  """
  callback=None
  bounds=None
  # current_activate_index: indices of the currently active basis functions
  # list/array of basis-term indices that are currently active in the optimization
  # this is the current constitutive model hypothesis being tested (subset of basis functions
  # of the candidate library)
  current_activate_index=args_dict['activate_basis_index']
  # target_index: index of the target vector
  target_index=args_dict['target_vector_index']
  # num_theta: number of theta values. Total number of candidate terms (size of library)
  num_theta=args_dict['num_theta']
  # num_base_orign: number of basis functions
  num_base_orign=len(current_activate_index)
  # meshName: name of the mesh, used to load the mesh
  meshName = args_dict['meshName']
  # tendonStamp: date stamp of the tendon, used to identify the specific sample
  tendonStamp = args_dict['tendonStamp']
  # combination: combination of basis functions
  combination = args_dict['combination']
  # lossFactor: factor to scale the loss function, used to adjust the weight of the loss function
  lossFactor = args_dict['lossFactor']
  
  # frozen_index: indices you are not allowed to drop (always keep these terms)
  frozen_index=[]
  # max_eliminate_step: maximum number of elimination allowed
  max_eliminate_step=num_base_orign-1
  if 'frozen_index' in args_dict.keys():
    frozen_index=args_dict['frozen_index']
  if 'max_eliminate_step' in args_dict.keys():
    max_eliminate_step=args_dict['max_eliminate_step']
  if 'method' in args_dict.keys():
    # method: scipy optimization method (e.g., 'L-BFGS-B', 'TNC', etc.)
    # L-BFGS-B is a quasi-Newton method that approximates the Broyden-Fletcher-Goldfarb-Shanno (BFGS)
    # algorithm using a limited amount of computer memory. It is particularly well-suited for 
    # large-scale optimization problems.
    method=args_dict['method']
  if 'bounds' in args_dict.keys():
    # bounds: coefficient bounds per active parameter (needed for L-BFGS-B and TNC methods)
    bounds=args_dict['bounds']
    
  if 'disp' in method_options.keys():
    if method_options['disp']==True:
      callback=iter_cb
  # gamma_matrix: matrix to store the coefficients of the basis functions at each step
  # shape: (num_theta, max_eliminate_step+1)
  gamma_matrix=np.empty((num_theta,max_eliminate_step+1))
  gamma_matrix[:]=np.NaN
  # the target coefficient is always 1
  # typical in VSI to fix one coefficient to 1 to avoid trivial rescaling ambiguity
  gamma_matrix[target_index,:]=1
  
  # if save_to_file is provided, open the file for appending results
  if save_to_file!=None:
    f=open(save_to_file,'ab')
  
  # loss: array to store the loss function value at each step
  loss=np.zeros(max_eliminate_step+1)

  # Step 0: optimize using the full initial set of active basis functions
  res=sp.minimize(obj_f, x0,jac=grad_f, method=method, 
                  args=(tendonStamp, meshName, lossFactor, combination,
                        current_activate_index,target_index),
                        bounds=bounds, options=method_options,callback=callback )
  x0=res.x
  gamma_matrix[current_activate_index,0]=x0
  loss[0]=res.fun
  print('==============================================================')
  print('step=',0, ' current_activate_index=',current_activate_index,
        ' x0=',gamma_matrix[:,0],' loss=',loss[0] )
  if save_to_file!=None:
    info=np.reshape(np.append(gamma_matrix[:,0],(loss[0])),(1,-1))
    np.savetxt(f,info)
    f.flush()
    os.fsync(f.fileno())

  for step in range(len(current_activate_index)-1):
    if step==max_eliminate_step:
      break
    num_activate_index=len(current_activate_index)
    gamma_matrix_tem=np.zeros((num_activate_index-1,num_activate_index))
    loss_tem=np.ones(num_activate_index)*1.0e20 # why*1e20?
    for j in range(len(current_activate_index)):
      try_index=current_activate_index[j]
      # continue if j is in the frozen_index
      if try_index in frozen_index:
        continue
        
      current_activate_index_tem=np.delete(current_activate_index,j)
      x0_tem=np.delete(x0,j)
      bounds_tem=None
      if 'bounds' in args_dict.keys():
        bounds_tem=np.delete(bounds,j,0)
      res=sp.minimize(obj_f, x0_tem,jac=grad_f,  method=method, 
                      args=(tendonStamp, meshName, lossFactor, combination,
                            current_activate_index_tem,target_index),
                      bounds=bounds_tem, 
                      options=method_options,callback=callback)
      gamma_matrix_tem[:,j]=res.x
      loss_tem[j]=res.fun
    
    drop_index=np.argmin(loss_tem) 
    print('loss_try=',loss_tem)
    loss_try=loss_tem[drop_index]  
    F=(loss_try-loss[step])/loss[step]*(num_base_orign-num_activate_index+1)
    
    if F<F_threshold:
      current_activate_index=np.delete(current_activate_index,drop_index)
      x0=np.delete(x0,drop_index)
      if 'bounds' in args_dict.keys():
        bounds=np.delete(bounds,drop_index,0)
      gamma_matrix[current_activate_index,step+1]=gamma_matrix_tem[:,drop_index]
      loss[step+1]=loss_try
      
    else:
      break
    if save_to_file!=None:
      info=np.reshape(np.append(gamma_matrix[:,step+1],(loss[step+1])),(1,-1))
      np.savetxt(f,info)
      f.flush()
      os.fsync(f.fileno())
    print('==============================================================')
    print('step=',step+1, ' current_activate_index=',current_activate_index,' x0=',gamma_matrix[:,step+1],' loss=',loss[step+1] )
    
  return gamma_matrix, loss