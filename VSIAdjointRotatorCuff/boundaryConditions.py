"""
The code below creates a dictionary defining the boundary conditions for the rotator cuff 
tendon model for both intact and torn tendon cases. They keys are the tendon conditions 
and the values are the boundary conditions for each facet. The boundary conditions are 
defined as a lambda function that returns True if the facet satisfies the boundary condition 
and False otherwise.
"""

Tendon20231012 = {
  # key 'equations' maps to another disctionary that are the boundary classifications for 
  # each facet
  "equations": {
    # boundary conditions for intact with 3 boundary "labels" "tendon", "enthesis", and "head"
    "Intact": {
      """
      "lambda f" defines an anonymous function that takes a facet "f" as input
      "f.normal(i)" returns the i-th component of the normal vector of the facet
      "f.normal(0)" is the x-component of the normal vector
      "f.exterior()" returns True if the facet is on the exterior of the mesh
      "f.midpoint()[i]" returns the i-th component of the midpoint of the facet
      "f.normal <-0.8" means the normal vector is pointing mostly in the negative x-direction
      "f.midpoint()[0]<-15." means the x-coordinate of the midpoint is less than -15
      
      Net effect: “tendon” boundary ≈ exterior facets far in −x, with normals mostly pointing −x.
      """

      "tendon": lambda f: f.normal(0)<-0.8 and f.exterior() and f.midpoint()[0]<-15.,
      
      """
      "f.normal(1)" is the y-component of the normal vector
      "f.normal(1)<-0.7" means the normal vector is pointing mostly in the negative y-direction
      "f.exterior()" returns True if the facet is on the exterior of the mesh
      "f.midpoint()[0]- 3.18e-4 
      * f.midpoint()[2]**4 + 0.0033 
      * f.midpoint()[2]**3 - 0.0023 
      * f.midpoint()[2]**2 - 0.3170 
      * f.midpoint()[2] - 0.7776 > 0." 
      is a geometric seperator curve/surface in the x-z plane.
      
      Net effect: “enthesis” facets are exterior, face −y, 
      and lie on one side of a fitted quartic boundary in x–z.
      """

      "enthesis": lambda f: f.normal(1)<-0.7 and f.exterior() and \
        f.midpoint()[0]-3.18e-4*f.midpoint()[2]**4+0.0033*f.midpoint()[2]**3-\
            0.0023*f.midpoint()[2]**2-0.3170*f.midpoint()[2]-0.7776 > 0.,
      
      """
      "f.normal(1)" is the y-component of the normal vector
      "f.normal(1)<-0.3" means the normal vector is pointing mostly in the negative y-direction
      "f.exterior()" returns True if the facet is on the exterior of the mesh
      "f.midpoint()[0] is the opposite of the "enthesis" condition, defining the "head" region.
      So it selects the facts on the other side of the geometric seperator curve/surface.
      Net effect: “head” facets are exterior, face −y, locate d between two fitted boundaries in x–z.
      """

      "head": lambda f: f.normal(1)<-0.3 and f.exterior() and \
        f.midpoint()[0]-3.18e-4*f.midpoint()[2]**4+0.0033*f.midpoint()[2]**3-\
            0.0023*f.midpoint()[2]**2-0.3170*f.midpoint()[2]-0.7776 < 0. and \
                f.midpoint()[0]-0.0551*f.midpoint()[2]**2-\
                    0.2501*f.midpoint()[2]+13.8244 > 0.
    },

    "Torn": {
        
    """
    Same as the Intact tendon rule
    """

    "tendon": lambda f: f.normal(0)<-0.8 and f.exterior() and f.midpoint()[0]<-15.,
    """
    Two conditions for the "enthesis" boundary in the torn tendon case.
    and (
    A < 0. 
    or 
    B > 0.
    )
    where condition A is:
    f.midpoint()[0]-0.0010*f.midpoint()[2]**4
    -0.0139*f.midpoint()[2]**3 - 0.1176*f.midpoint()[2]**2 
    - 0.3674*f.midpoint()[2] - 3.4498 < 0.
    
    and condition B is:
    f.midpoint()[2]+0.0009*f.midpoint()[0]**3+
    0.1801*f.midpoint()[0]**2
    -3.2451*f.midpoint()[0] + 9.6708 > 0.

    Condition A is another fitted quartic boundary in x–z plane.
    Condition B is a cubic boundary in z–x plane.
    Net effect: in the torn case, “enthesis” is a more carefully clipped region, 
    presumably to exclude the torn/open boundary or separate remaining attachment.
    """

    "enthesis": lambda f: f.normal(1)<-0.7 and f.exterior() and \
      f.midpoint()[0]-3.18e-4*f.midpoint()[2]**4+0.0033*f.midpoint()[2]**3-\
        0.0023*f.midpoint()[2]**2-0.3170*f.midpoint()[2]-0.7776 > 0. \
          and (f.midpoint()[0]-0.0010*f.midpoint()[2]**4\
            -0.0139*f.midpoint()[2]**3 - 0.1176*f.midpoint()[2]**2 \
                - 0.3674*f.midpoint()[2] - 3.4498 < 0. or \
                    f.midpoint()[2]+0.0009*f.midpoint()[0]**3+\
                        0.1801*f.midpoint()[0]**2\
                        -3.2451*f.midpoint()[0] + 9.6708 > 0.),

    """
    Similar to the "head" condition in the intact case with slight key change.
    "f.normal(1)<-0.5" instead of "<-0.3", making the normal vector strictly 
    more negative in y-direction.

    """

    "head": lambda f: f.normal(1)<-0.5 and f.exterior() and \
      f.midpoint()[0]-3.18e-4*f.midpoint()[2]**4+0.0033*f.midpoint()[2]**3-\
        0.0023*f.midpoint()[2]**2-0.3170*f.midpoint()[2]-0.7776 < 0. and \
            f.midpoint()[0]-0.0551*f.midpoint()[2]**2-\
              0.2501*f.midpoint()[2]+13.8244 > 0.
    }
  }
}

"""
What f.normal, f.exterior, and f.midpoint are in practice:
These lambdas assume f is a boundary facet from FEniCS mesh iteration / marking.

f.exterior() → True if the facet is on the external boundary
f.midpoint() → returns a 3D point-like object; [0],[1],[2] correspond to x,y,z
f.normal(i) → i-th component of the outward unit normal

How are the values chosen?
f.exterior(), f.normal(i), and f.midpoint() are not chosen by the user.
They are geometric quantities computed automatically by FEniCS for each boundary facet.
What is chosen by the user are the numerical thresholds and polynomial expressions used 
to select subsets of those facets.

f is a mesh facet (a boundary face in 3D, or an edge in 2D)
Somewhere in the other codes FEniCS iterates over all boundary facets f of the mesh.
for f in facets(mesh):
    if f.exterior():
        ...
f.exterion(), if True, indicates f is on the outer boundary of the mesh. If false,
it is an interior facet (between two elements inside the mesh).
  Used to apply boundary conditions only on the outer surface. Like traction boundary conditions 
  or Dirichlet BCs.

f. midpoint() returns a point object representing the geometric center of the facet.
 for examplem, in a 3D triangle facet, it is the average of the three vertex coordinates.

f.normal(i) returns the i-th component of the outward unit normal vector to the facet.
 For exmaple, f.normal(0) is the x-component of the normal vector. f.normal(1) is the y-component.
 and f.normal(2) is the z-component.

What is chosen by the user?
The thresholds and inequalities applied to these quantities.
For example, f.normal(0)<-0.8 means the x-component of the normal vector is less than -0.8,
  Selects facts whose outward normal points mostly in the negative x-direction.
  Since normals are unit vectors:
  -1 = perfectly negative x-direction
  1 = perfectly positive x-direction
  -0.8 ~ within 36 degrees of negative x-direction

How are the numerical thresholds chosen?
Visualizing facet normals in Paraview or other visualization software.
Example: f.normal(i) < -0.9, very strict, nearly planar surface facing -i direction
Example: f.normal(i) < -0.3, looser, allows curvature and facets that are not perfectly aligned.

For midpoint-based conditions:
Come from fitted curves/surfaces to anatomical boundaries in the x-z plane.
Typically by exporting the mesh or MATLAB, manually identifying boundary points, fitting a polynomial,
embedding that polynomial as a classifier in the code.

Take away:
each lambda is asking:
Does this facet (a) lie on the exterior boundary,
(b) face roughly this direction, and (c) lie on this side of a geometric boundary in space?
"""