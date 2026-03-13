# meshioConvertMesh.py
from pathlib import Path
import meshio

# Create mesh and define function space
# cartilageList = ["20250808", "20250811", "20250812", "20250816"]
# folderList = [
#             ["humanIntact101","humanIntact201"], 
#             ["humanIntact401","humanIntactC01","humanIOCABinA01"],
#             ["humanOCABinD01"],
#             ["humanOCABinC_V201"]
#             ]
# base = Path(".")
# for date, subfolders in zip(cartilageList, folderList):
#     date_dir = base / date
#     for patella in subfolders:
#         sub_dir = date_dir / patella / "DataProcessFiles"
#         inp_file = sub_dir / f"Patella{date}_{patella}_Mask_Smoothed_INP.inp"
#         xdmf_file = sub_dir / f"Patella{date}_{patella}_Mask_Smoothed_INP.xdmf"
#         mesh = meshio.read(inp_file)
#         meshio.write(xdmf_file, mesh)
#         print(f"Converted {inp_file} → {xdmf_file} successfully.")

base = Path(".")
inp_file = base / "VSI-HE-Test" / "Strain_0p8" / "HE_Confined_0p8.inp"
xdmf_file = base / "VSI-HE-Test" / "Strain_0p8" / "cube_mesh.xdmf"
mesh = meshio.read(inp_file)
meshio.write(xdmf_file, mesh)
print(f"Converted {inp_file} → {xdmf_file} successfully.")
