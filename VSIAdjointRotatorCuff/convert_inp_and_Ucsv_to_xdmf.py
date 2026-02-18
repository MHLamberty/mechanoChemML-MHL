from pathlib import Path
import numpy as np
import meshio


def read_abaqus_nodes_from_inp(inp_path: Path):
    """
    Parse the *NODE section of an Abaqus .inp to get:
      node_label -> (x,y,z)
    """
    node_coords = {}
    in_node_section = False

    with inp_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            # Start of node section
            if s.upper().startswith("*NODE"):
                in_node_section = True
                continue

            # Any new keyword ends the node section
            if in_node_section and s.startswith("*"):
                break

            if in_node_section:
                # Format: label, x, y, z  (sometimes spaces)
                parts = [p.strip() for p in s.split(",")]
                if len(parts) >= 4:
                    try:
                        lab = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                        node_coords[lab] = (x, y, z)
                    except ValueError:
                        # skip malformed lines
                        pass

    if not node_coords:
        raise RuntimeError(f"Could not find *NODE section in {inp_path}")
    return node_coords


def read_cube_U_csv(csv_path: Path):
    """
    cube_U.csv must have columns:
      nodeLabel,U1,U2,U3
    Returns dict: node_label -> (U1,U2,U3)
    """
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding=None)
    # Handle case where only one row exists (genfromtxt returns 0-d structured array)
    if data.shape == ():
        data = np.array([data])

    u_by_label = {}
    for row in data:
        lab = int(row["nodeLabel"])
        u_by_label[lab] = (float(row["U1"]), float(row["U2"]), float(row["U3"]))
    return u_by_label


def coord_key(xyz, tol=1e-9):
    """
    Create a hashable key for a coordinate with tolerance.
    We round to the nearest multiple of tol.
    """
    return tuple((np.round(np.array(xyz) / tol) * tol).tolist())


def attach_displacements(mesh: meshio.Mesh, inp_nodes: dict, u_by_label: dict, tol=1e-9):
    """
    Map nodeLabel-based displacements onto mesh.points by matching coordinates.

    This works even if meshio reorders points, as long as coordinates match.
    """
    # Build a map from coordinate -> nodeLabel
    coord_to_label = {}
    for lab, xyz in inp_nodes.items():
        coord_to_label[coord_key(xyz, tol=tol)] = lab

    U = np.zeros((mesh.points.shape[0], 3), dtype=float)
    missing = 0

    for i, p in enumerate(mesh.points):
        k = coord_key(p, tol=tol)
        lab = coord_to_label.get(k, None)
        if lab is None:
            missing += 1
            continue
        u = u_by_label.get(lab, (0.0, 0.0, 0.0))
        U[i, :] = u

    if missing > 0:
        raise RuntimeError(
            f"Could not match {missing} mesh points to INP node coordinates. "
            f"Try increasing tol (e.g., tol=1e-6) or confirm units/format."
        )

    mesh.point_data["U"] = U
    return mesh


def main():
    base = Path(".")
    inp_file = base / "VSI-Test" / "TestVSI-Tets.inp"
    u_csv = base / "VSI-Test" / "cube_U.csv"
    xdmf_file = base / "VSI-Test" / "cube_with_U.xdmf"

    print(f"Reading mesh: {inp_file}")
    mesh = meshio.read(inp_file)

    print(f"Parsing INP nodes for coordinate mapping...")
    inp_nodes = read_abaqus_nodes_from_inp(inp_file)

    print(f"Reading displacements: {u_csv}")
    u_by_label = read_cube_U_csv(u_csv)

    print("Attaching displacement field to mesh point_data as 'U'...")
    # For mm-scale meshes, tol=1e-9 is usually fine; if you get mismatch, try 1e-6
    mesh = attach_displacements(mesh, inp_nodes, u_by_label, tol=1e-9)

    print(f"Writing XDMF: {xdmf_file}")
    meshio.write(xdmf_file, mesh)

    print(f"Done. Wrote mesh + point field U to: {xdmf_file}")
    print("You should also see a companion HDF5 file in the same folder.")


if __name__ == "__main__":
    main()
