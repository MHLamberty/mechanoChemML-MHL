import meshio

m = meshio.read("HE_Unconfined copy.inp")

print("Read OK.")
print("cell block types:", [c.type for c in m.cells])
print("cells_dict keys:", list(m.cells_dict.keys()))