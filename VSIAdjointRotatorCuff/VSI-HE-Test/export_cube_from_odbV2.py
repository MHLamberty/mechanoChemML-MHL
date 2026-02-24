# export_cube_from_odb.py
from odbAccess import openOdb
import os

print("CWB =", os.getcwd())

case_tag = "HE_SimpleShear"          # label outputs
odb_name = "HE_SimpleShear.odb"      # job odb
step_name = "Compression"              # make consistent across jobs if possible

# Face node sets (must exist as NODE SETS)
set_top = "YMAX"
set_bot = "YMIN"

# Which RF component to export:
# 0 -> RF1 (x), 1 -> RF2 (y), 2 -> RF3 (z)
rf_comp = 0

u_out = "cube_U_" + case_tag + ".csv"
rf_out = "cube_RF" + str(rf_comp + 1) + "_" + case_tag + ".txt"
out_path = os.path.abspath(rf_out)

print("ODB =", odb_name)
print("STEP =", step_name)
print("TOP SET =", set_top, "BOT SET =", set_bot, "RF component =", rf_comp + 1)
print("Writing RF to:", out_path)

odb = openOdb(odb_name)
step = odb.steps[step_name]
frame = step.frames[-1]

asm = odb.rootAssembly
# If instance name differs, update here
inst = asm.instances["CUBE-1"]

# --- Displacements ---
U = frame.fieldOutputs["U"]
nodes = inst.nodes

u_by_label = {}
for v in U.values:
    if v.instance.name == inst.name:
        u_by_label[v.nodeLabel] = v.data

f = open(u_out, "w")
f.write("nodeLabel,U1,U2,U3\n")
for n in sorted([nd.label for nd in nodes]):
    u = u_by_label.get(n, (0.0, 0.0, 0.0))
    f.write("{},{},{},{}\n".format(n, u[0], u[1], u[2]))
f.close()

# --- Reaction forces ---
RF = frame.fieldOutputs["RF"]

def get_nodeset(nodeset_name):
    key = nodeset_name.upper()
    if key in asm.nodeSets:
        return asm.nodeSets[key]
    if key in inst.nodeSets:
        return inst.nodeSets[key]
    raise RuntimeError("Node set not found: " + nodeset_name)

def sum_rf_on_nodeset(nodeset_name, comp):
    ns = get_nodeset(nodeset_name)
    s = 0.0
    subset = RF.getSubset(region=ns)
    for v in subset.values:
        s += v.data[comp]
    return s

rf_top = sum_rf_on_nodeset(set_top, rf_comp)
rf_bot = sum_rf_on_nodeset(set_bot, rf_comp)

f = open(out_path, "w")
f.write("RF{}_top = {}\n".format(rf_comp + 1, rf_top))
f.write("RF{}_bot = {}\n".format(rf_comp + 1, rf_bot))
f.close()

odb.close()

print("Wrote " + u_out + " and " + rf_out)