from pathlib import Path
import meshio

def inp_to_xdmf_mesh(inp_path: Path, xdmf_path: Path):
    msh = meshio.read(inp_path)

    # Keep only tetra cells (common for C3D4/C3D4H exports)
    tetra_cells = None
    for cell_block in msh.cells:
        if cell_block.type in ("tetra", "tetra10"):
            tetra_cells = cell_block
            break
    if tetra_cells is None:
        raise RuntimeError("No tetra cells found in INP: " + str([c.type for c in msh.cells]))

    msh_out = meshio.Mesh(points=msh.points, cells=[tetra_cells])
    meshio.write(xdmf_path, msh_out)
    print("Wrote mesh XDMF:", xdmf_path)

if __name__ == "__main__":
    base = Path(".")
    inp_file = base / "VSI-HE-Test" / "HE_Unconfined copy.inp"
    xdmf_file = base / "VSI-HE-Test" / "cube_mesh.xdmf"
    inp_to_xdmf_mesh(inp_file, xdmf_file)