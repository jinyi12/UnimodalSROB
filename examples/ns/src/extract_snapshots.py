import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path


def extract_snapshots(data, start_time, end_time, num_snapshots, dt):
    start_idx = int(start_time / dt)
    end_idx = int(end_time / dt)
    total_steps = end_idx - start_idx + 1

    # Calculate the stride to get num_snapshots
    stride = max(1, total_steps // num_snapshots)

    # Extract snapshots with the calculated stride
    snapshots = data[start_idx : end_idx + 1 : stride]

    # If we have more than num_snapshots, truncate
    snapshots = snapshots[:num_snapshots]

    snapshots_2d = snapshots[:, :, :2]
    n_timesteps = snapshots_2d.shape[0]
    u_component = snapshots_2d[:, :, 0].reshape(n_timesteps, -1).T
    v_component = snapshots_2d[:, :, 1].reshape(n_timesteps, -1).T
    return np.vstack((u_component, v_component))


@hydra.main(
    version_base=None, config_path="../conf", config_name="extract_snapshots_config"
)
def main(cfg: DictConfig):
    base_path = Path(cfg.base_data_path)
    output_base_path = Path(cfg.output_base_path)

    if cfg.mode == "train":
        mu_values = cfg.train_mu_values
        t_start = cfg.t_start_train
        t_end = cfg.t_end_train
        num_snapshots = cfg.num_train_snapshots
    elif cfg.mode == "test":
        mu_values = [cfg.test_mu_value]
        t_start = cfg.t_start_test
        t_end = cfg.t_end_test

        t_train_interval = cfg.t_end_train - cfg.t_start_train
        cfg.num_test_snapshots = int(
            (t_end - t_start) / t_train_interval * cfg.num_train_snapshots
        )
        num_snapshots = cfg.num_test_snapshots
    else:
        raise ValueError("Invalid mode. Choose 'train' or 'test'.")

    for mu in mu_values:
        Re = int(np.round(cfg.U_c * cfg.L_c * cfg.rho / mu))
        folder_name = f"Re_{Re}_mu_{mu:.6f}"
        folder_path = base_path / folder_name
        input_file = folder_path / "u.npy"

        if not input_file.exists():
            print(f"No u.npy file found at {input_file}. Skipping...")
            continue

        output_folder = output_base_path / folder_name
        output_folder.mkdir(parents=True, exist_ok=True)
        output_file = output_folder / f"u_snapshots_matrix_{cfg.mode}.npy"

        full_data = np.load(input_file)
        snapshot_matrix = extract_snapshots(
            full_data, t_start, t_end, num_snapshots, cfg.dt
        )

        np.save(output_file, snapshot_matrix)

        print(f"Saved snapshot matrix to {output_file}")
        print(f"Shape of snapshot matrix: {snapshot_matrix.shape}")

        # Calculate and save the effective dt
        effective_dt = (t_end - t_start) / (num_snapshots)

        # Add effective_dt to the config
        cfg.effective_dt = effective_dt

        metadata = OmegaConf.to_yaml(cfg)
        metadata_path = output_folder / f"snapshots_matrix_{cfg.mode}_metadata.yaml"
        with open(metadata_path, "w") as f:
            f.write(metadata)
        print(f"Saved metadata to {metadata_path}")
        print(f"Effective dt: {effective_dt}")


if __name__ == "__main__":
    main()
