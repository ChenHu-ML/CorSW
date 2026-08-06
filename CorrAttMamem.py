import os
import glob
import torch
import numpy as np
from utils.functions import resolve_device, save_results, set_seed, train_model
from models.CorAtt import CorrAttMamem
from utils.GetMamem import get_all_dataloader
import argparse
import hydra
from omegaconf import DictConfig


def get_args(cfg: DictConfig) -> argparse.Namespace:
    num_sessions = int(cfg.num_sessions)
    test_session = int(cfg.test_session)
    val_session = int(cfg.val_session)
    if val_session <= 0:
        val_session = ((test_session - 2) % num_sessions) + 1

    args = argparse.Namespace(
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
        num_sessions=num_sessions,
        test_session=test_session,
        val_session=val_session,
        swd_weight=float(cfg.swd_weight),
        swd_num_projections=int(cfg.swd_num_projections),
        swd_p=float(cfg.swd_p),
        swd_random_state=int(cfg.swd_random_state),
        swd_metric=str(cfg.swd_metric),
        swd_input_space=str(cfg.swd_input_space),
    )
    return args


def save_run_result(cfg: DictConfig, args: argparse.Namespace, accuracies: np.ndarray) -> None:
    run_root = str(cfg.run_results_dir)
    results_dir = hydra.utils.to_absolute_path(run_root)
    os.makedirs(results_dir, exist_ok=True)
    run_path = os.path.join(
        results_dir,
        f'run_sub{args.sub:02d}_testS{args.test_session}_valS{args.val_session}.npz',
    )
    np.savez(
        run_path,
        subject=args.sub,
        test_session=args.test_session,
        val_session=args.val_session,
        accuracies=accuracies,
    )


def aggregate_results(cfg: DictConfig) -> None:
    run_root = str(cfg.run_results_dir)
    results_dir = hydra.utils.to_absolute_path(run_root)
    num_subjects = int(cfg.num_subjects)
    num_sessions = int(cfg.num_sessions)
    num_repeats = int(cfg.num_repeats)
    model_name = str(cfg.model_name)
    res_path = hydra.utils.to_absolute_path(str(cfg.res_path))

    results = np.full((num_subjects, num_sessions, num_repeats), np.nan, dtype=np.float32)
    files = sorted(glob.glob(os.path.join(results_dir, "run_sub*_testS*_valS*.npz")))
    for path in files:
        data = np.load(path)
        subject = int(data["subject"])
        test_session = int(data["test_session"])
        accuracies = np.asarray(data["accuracies"], dtype=np.float32)
        max_reps = min(num_repeats, accuracies.shape[0])
        results[subject - 1, test_session - 1, :max_reps] = accuracies[:max_reps]

    fold_means = []
    fold_stds = []
    fold_labels = []
    for fold_idx in range(num_sessions):
        fold_matrix = results[:, fold_idx, :]
        fold_tensor = torch.tensor(fold_matrix)
        val_session = ((fold_idx + 1 - 2) % num_sessions) + 1
        fold_path = os.path.join(
            res_path,
            f'sessT{fold_idx + 1}_V{val_session}_in_size{cfg.in_size}_out_size{cfg.out_size}_epochs{cfg.epochs}'
            f'_lr{cfg.lr}_wd{cfg.wd}_{cfg.optim}_metric{cfg.metric}.txt',
        )
        save_results(fold_tensor, fold_path)

        repeat_means = np.nanmean(fold_matrix, axis=0)
        fold_mean = float(np.nanmean(repeat_means))
        fold_std = float(np.nanstd(repeat_means, ddof=0))
        fold_means.append(fold_mean)
        fold_stds.append(fold_std)
        train_sessions = [s for s in range(1, num_sessions + 1) if s not in (fold_idx + 1, val_session)]
        train_label = ''.join(str(s) for s in train_sessions)
        fold_labels.append(f'S{train_label}->S{fold_idx + 1} (val S{val_session})')

    overall_mean = float(np.nanmean(fold_means))
    latex_cells = [f'{mean:.2f} $\\pm$ {std:.2f}' for mean, std in zip(fold_means, fold_stds)]
    latex_line = f'{model_name} & ' + ' & '.join(latex_cells) + f' & {overall_mean:.2f} \\\\'
    summary_path = os.path.join(
        res_path,
        f'session_folds_in_size{cfg.in_size}_out_size{cfg.out_size}_epochs{cfg.epochs}'
        f'_lr{cfg.lr}_wd{cfg.wd}_{cfg.optim}_metric{cfg.metric}_summary.txt',
    )
    summary_lines = [
        f'Mean: {overall_mean:.4f}',
        f'Fold.Mean: {", ".join([f"{x:.4f}" for x in fold_means])}',
        f'Fold.Std:  {", ".join([f"{x:.4f}" for x in fold_stds])}',
        f'Fold.Labels: {"; ".join(fold_labels)}',
        f'Latex: {latex_line}',
    ]
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines) + '\n')


@hydra.main(config_path="./conf/", config_name="mamem.yaml", version_base=None)
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
        model = CorrAttMamem(args)
        accuracy = train_model(model, train_loader, valid_loader, test_loader, args)
        accuracy_pct = accuracy * 100
        print(f'{accuracy_pct:.2f}')
        accuracies.append(accuracy_pct)
    save_run_result(cfg, args, np.array(accuracies, dtype=np.float32))


if __name__ == '__main__':
    main()
