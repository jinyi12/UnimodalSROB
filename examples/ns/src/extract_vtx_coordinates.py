import adios2
from adios2 import Stream
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import adios4dolfinx


def read_adios2_data(file_path):
    with Stream(file_path, "r") as fh:
        for step in fh:
            geometry = step.read("geometry")
            connectivity = step.read("connectivity")
            u = step.read("u")
            return geometry, connectivity, u


import dolfinx
from mpi4py import MPI
from petsc4py import PETSc


def inspect_bp_file(filename):
    with Stream(filename, "r") as fh:
        print(f"File: {filename}")
        print("\nAvailable variables:")
        for name, info in fh.available_variables().items():
            print(f"  {name}:")
            for key, value in info.items():
                print(f"    {key}: {value}")

        print("\nAttributes:")
        for name, value in fh.available_attributes().items():
            print(f"  {name}: {value}")

        print("\nContents:")
        for step in fh:
            print(f"Step {step.current_step()}:")
            for name in fh.available_variables().keys():
                data = step.read(name)
                print(f"  {name}:")
                print(f"    Shape: {data.shape}")
                print(f"    Type: {data.dtype}")
                if data.size < 10:  # Print small arrays entirely
                    print(f"    Data: {data}")
                else:  # Print first few elements of larger arrays
                    print(f"    First few elements: {data.flatten()[:5]}...")


def read_and_print_original_point_ids(file_path):
    with Stream(file_path, "r") as fh:
        for step in fh:
            # Try to read vtkOriginalPointIds
            try:
                original_point_ids = step.read("vtkOriginalPointIds")
                has_original_ids = True
            except ValueError:
                has_original_ids = False

            # Read geometry for reference
            geometry = step.read("geometry")

            if has_original_ids:
                print("Original Point IDs:")
                for i, (orig_id, coord) in enumerate(zip(original_point_ids, geometry)):
                    print(
                        f"Current ID: {i}, Original ID: {orig_id}, Coordinates: {coord}"
                    )
            else:
                print("vtkOriginalPointIds not found in the file.")
                print("Current point IDs are:")
                for i, coord in enumerate(geometry):
                    print(f"Point ID: {i}, Coordinates: {coord}")

            # We only process the first step
            break


def print_mesh_info(mesh: dolfinx.mesh.Mesh):
    cell_map = mesh.topology.index_map(mesh.topology.dim)
    node_map = mesh.geometry.index_map()
    print(
        f"Rank {mesh.comm.rank}: number of owned cells {cell_map.size_local}",
        f", number of ghosted cells {cell_map.num_ghosts}\n",
        f"Number of owned nodes {node_map.size_local}",
        f", number of ghosted nodes {node_map.num_ghosts}",
    )


def read_mesh(filename: Path):
    from mpi4py import MPI

    import dolfinx

    import adios4dolfinx

    mesh = adios4dolfinx.read_mesh(
        filename,
        comm=MPI.COMM_WORLD,
        engine="BP4",
        ghost_mode=dolfinx.mesh.GhostMode.none,
    )
    print_mesh_info(mesh)

    return mesh


@hydra.main(version_base=None, config_path="../conf", config_name="extract_config")
def main(cfg: DictConfig):
    # Ensure base_data_path is provided
    if cfg.base_data_path == "???":
        raise ValueError("base_data_path must be provided via command line")

    base_path = Path(cfg.base_data_path)
    output_base_path = Path(cfg.output_base_path)

    for mu in cfg.mu_values:
        # Calculate Reynolds number
        Re = int(np.round(cfg.U_c * cfg.L_c * cfg.rho / mu))

        # Construct folder name
        folder_name = f"Re_{Re}_mu_{mu:.6f}"
        folder_path = base_path / folder_name

        # Construct .bp file path
        bp_file_path = folder_path / (cfg.bp_filename_prefix + f"_mu{mu:.6f}.bp")

        if not bp_file_path.exists():
            print(f"No .bp file found at {bp_file_path}. Skipping...")
            continue

        # Construct output path
        output_folder = output_base_path / folder_name
        output_folder.mkdir(parents=True, exist_ok=True)
        output_file = output_folder / "mesh_data.npz"

        # Read the data
        coords, connectivity, u = read_adios2_data(str(bp_file_path))
        # inspect_bp_file(str(bp_file_path))
        # read_and_print_original_point_ids(str(bp_file_path))
        read_mesh(str(bp_file_path))

        # Save coordinates and connectivity
        np.savez(output_file, coords=coords, connectivity=connectivity)

        print(f"Saved coordinates and connectivity to {output_file}")
        print(f"Shape of coords: {coords.shape}")
        print(f"Shape of connectivity: {connectivity.shape}")
        print(f"Shape of u: {u.shape}")

        # Save metadata
        metadata = OmegaConf.to_yaml(cfg)
        metadata_path = output_folder / "mesh_metadata.yaml"
        with open(metadata_path, "w") as f:
            f.write(metadata)
        print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
