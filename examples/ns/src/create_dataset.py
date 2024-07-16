import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path


@hydra.main(
    version_base=None, config_path="../conf", config_name="create_dataset_config"
)
def main(cfg: DictConfig):

    base_path = Path(cfg.base_data_path)
    output_path = Path(cfg.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = []

    for mu in cfg.mu_values:
        # Calculate Reynolds number
        Re = int(np.round(cfg.U_c * cfg.L_c * cfg.rho / mu))

        # Construct folder name
        folder_name = f"Re_{Re}_mu_{mu:.6f}"
        folder_path = base_path / folder_name

        # Construct input file path
        input_file = folder_path / "u_snapshots_matrix_train.npy"

        if not input_file.exists():
            print(f"No file found at {input_file}. Skipping...")
            continue

        # Load the snapshot matrix
        snapshot_matrix = np.load(input_file)

        # Append to the dataset list
        dataset.append(snapshot_matrix)

    # Convert list to numpy array
    combined_dataset = np.array(dataset)

    # Save the combined dataset
    output_file = output_path / cfg.output_file
    np.save(output_file, combined_dataset)

    print(f"Saved combined dataset to {output_file}")
    print(f"Shape of combined dataset: {combined_dataset.shape}")

    # Save metadata
    metadata = OmegaConf.to_yaml(cfg)
    metadata_path = output_path / "dataset_metadata.yaml"
    with open(metadata_path, "w") as f:
        f.write(metadata)
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
