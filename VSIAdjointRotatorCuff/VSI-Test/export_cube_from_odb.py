# export_cube_from_odb.py
from odbAccess import openOdb
import numpy as np

odb_name = "TestVSI-1.odb"       # change if needed
step_name = "Compression"
set_top = "TOPFACE"
set_bot = "BOTTOMFACE"
set_c1  = "CORNER1"
set_c2  = "CORNER 2"             # note: in the inp it's written as "Corner 2"
part_instance = "CUBE-1"         # instance name in assembly (often PARTNAME-1)

odb = openOdb(odb_name)
step = odb.steps[step_name]
frame = step.frames[-1]          # last frame

asm = odb.rootAssembly
inst = asm.instances[part_instance]

# --- Displacements ---
U = frame.fieldOutputs["U"]      # nodal displacement vector
# Get all nodes in the instance
nodes = inst.nodes
# Build a dict: nodeLabel -> displacement vector
u_by_label = {}
for v in U.values:
    if v.instance.name == part_instance:
        u_by_label[v.nodeLabel] = v.data  # (U1,U2,U3)

# Write nodal displacements for the whole mesh (sorted by node label)
with open("cube_U.csv", "w") as f:
    f.write("nodeLabel,U1,U2,U3\n")
    for n in sorted([nd.label for nd in nodes]):
        u = u_by_label.get(n, (0.0, 0.0, 0.0))
        f.write("{},{},{},{}\n".format(n, u[0], u[1], u[2]))

# --- Reaction forces ---
RF = frame.fieldOutputs["RF"]

def sum_rf2_on_nodeset(nodeset_name):
    # node set is stored on the assembly; use instance node set if needed
    # Try assembly-level first:
    if nodeset_name.upper() in asm.nodeSets:
        ns = asm.nodeSets[nodeset_name.upper()]
    else:
        # fall back to instance-level
        ns = inst.nodeSets[nodeset_name.upper()]
    rf2 = 0.0
    for v in RF.getSubset(region=ns).values:
        rf2 += v.data[1]         # RF2 (y-direction)
    return rf2

rf2_top = sum_rf2_on_nodeset(set_top)
rf2_bot = sum_rf2_on_nodeset(set_bot)

with open("cube_RF2.txt", "w") as f:
    print("RF2_top = {}".format(rf2_top))
    print("RF2_bot = {}".format(rf2_bot))

odb.close()
print("Wrote cube_U.csv and cube_RF2.txt")
