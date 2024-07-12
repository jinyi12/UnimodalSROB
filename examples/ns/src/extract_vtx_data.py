import adios2
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import os


def read_adios2_data(file_path, variable_name):
    with adios2.open(file_path, "r") as fh:
        steps = []
        for step in fh:
            var = step.read(variable_name)
            steps.append(var)
    return np.array(steps)


@hydra.main(version_base=None, config_path="../conf", config_name="extract_config")
def main(cfg: DictConfig):
    # Ensure base_data_path is provided
    if cfg.base_data_path == "???":
        raise ValueError("base_data_path must be provided via command line")

    base_path = Path(cfg.base_data_path)
    output_base_path = Path(cfg.output_base_path)

    for mu in cfg.mu_values:
        # Calculate Reynolds number, round to nearest integer
        Re = int(np.round(cfg.U_c * cfg.L_c * cfg.rho / mu))
        cfg.Re = Re
        cfg.num_steps = cfg.T / cfg.dt

        # Construct folder name
        folder_name = f"Re_{Re}_mu_{mu:.6f}"
        folder_path = base_path / folder_name

        # Find .bp folder
        # Construct .bp file path
        bp_file_path = folder_path / (cfg.bp_filename_prefix + f"_mu{mu:.6f}.bp")

        if not bp_file_path.exists():
            print(f"No .bp file found at {bp_file_path}. Skipping...")
            continue

        # Construct output path
        output_folder = output_base_path / folder_name
        output_folder.mkdir(parents=True, exist_ok=True)
        output_file = output_folder / f"{cfg.variable}.npy"

        # Read the data
        data = read_adios2_data(str(bp_file_path), cfg.variable)

        # Save the data
        np.save(output_file, data)

        print(f"Saved data to {output_file}")
        print(f"Number of time steps: {data.shape[0]}")
        print(f"Shape of data: {data.shape}")

        # Save metadata
        metadata = OmegaConf.to_yaml(cfg)
        metadata_path = output_folder / "metadata.yaml"
        with open(metadata_path, "w") as f:
            f.write(metadata)
        print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
