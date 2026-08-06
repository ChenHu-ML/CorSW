import os
import glob
import argparse

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from utils.functions import resolve_device, save_results, set_seed, train_model
from models.CorAtt import CorrAttCha
from utils.GetBCIcha import get_all_dataloader


def get_args(cfg: DictConfig) -> argparse.Namespace:
    return argparse.Namespace(
        sub=int(cfg.subject),
        lr=float(cfg.lr),
        wd=float(cfg.wd),
        iterations=int(cfg.iterations),
        epochs=int(cfg.epochs),
        bs=int(cfg.bs),
        model_path=hydra.utils.to_absolute_path(str(cfg.model_path)),
        data_path=hydra.utils.to_absolute_path(str(cfg.data_path)),
        res_path=hydra.utils.to_absolute_path(str(cfg.res_path)),
        scoring_metric=str(cfg.scoring_metric),
        optim=str(cfg.optim),
        device=resolve_device(cfg.device),
        in_size=int(cfg.in_size),
        out_size=int(cfg.out_size),
        metric=str(cfg.metric),
        nclass=int(cfg.nclass),
        num_sessions=int(cfg.num_sessions),
        test_session=int(cfg.test_session),
        val_session=int(cfg.val_session),
        swd_weight=float(cfg.swd_weight),
        swd_num_projections=int(cfg.swd_num_projections),
        swd_p=float(cfg.swd_p),
        swd_random_state=int(cfg.swd_random_state),
        swd_metric=str(cfg.swd_metric),
        swd_input_space=str(cfg.swd_input_space),
    )


def save_run_result(cfg: DictConfig, args: argparse.Namespace, accuracies: np.ndarray) -> None:
    run_root = str(cfg.run_results_dir)
    results_dir = hydra.utils.to_absolute_path(run_root)
    os.makedirs(results_dir, exist_ok=True)
    run_path = os.path.join(results_dir, f"run_sub{args.sub:02d}.npz")
    np.savez(run_path, subject=args.sub, accuracies=accuracies)


def aggregate_results(cfg: DictConfig) -> None:
    run_root = str(cfg.run_results_dir)
    results_dir = hydra.utils.to_absolute_path(run_root)
    num_repeats = int(cfg.num_repeats)
    subject_ids = [int(s) for s in cfg.subject_ids]

    results = np.full((len(subject_ids), num_repeats), np.nan, dtype=np.float32)
    subject_map = {sid: idx for idx, sid in enumerate(subject_ids)}

    files = sorted(glob.glob(os.path.join(results_dir, "run_sub*.npz")))
    for path in files:
        data = np.load(path)
        subject = int(data["subject"])
        accuracies = np.asarray(data["accuracies"], dtype=np.float32)
        row_idx = subject_map.get(subject)
        if row_idx is None:
            continue
        max_reps = min(num_repeats, accuracies.shape[0])
        results[row_idx, :max_reps] = accuracies[:max_reps]

    res_path = hydra.utils.to_absolute_path(str(cfg.res_path))
    os.makedirs(res_path, exist_ok=True)
    results_path = os.path.join(
        res_path,
        f"summary_in_size{cfg.in_size}_out_size{cfg.out_size}_epochs{cfg.epochs}"
        f"_lr{cfg.lr}_wd{cfg.wd}_{cfg.optim}_metric{cfg.metric}.txt",
    )
    save_results(torch.tensor(results), results_path)


@hydra.main(config_path="./conf/", config_name="bcicha.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    mode = str(cfg.mode).lower()
    if mode == "aggregate":
        aggregate_results(cfg)
        return
    if mode != "run":
        raise ValueError(f"Unsupported mode: {mode}")

    args = get_args(cfg)
    set_seed(int(cfg.seed))
    os.makedirs(args.res_path, exist_ok=True)

    num_repeats = int(cfg.num_repeats)
    accuracies = []
    for repeat_idx in range(num_repeats):
        args.repeat = repeat_idx + 1
        train_loader, valid_loader, test_loader = get_all_dataloader(args)
        model = CorrAttCha(args)
        accuracy = train_model(model, train_loader, valid_loader, test_loader, args)
        accuracy_pct = accuracy * 100
        print(f"{accuracy_pct:.2f}")
        accuracies.append(accuracy_pct)

    save_run_result(cfg, args, np.array(accuracies, dtype=np.float32))


if __name__ == "__main__":
    main()
