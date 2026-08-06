import os
import random
import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from torch.optim import Adam, Adadelta, AdamW, NAdam, RAdam, SGD
from tqdm import tqdm

from models import corsw

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested):
    """Resolve the optional automatic PyTorch device selection."""
    requested = str(requested)
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return requested


def create_optimizer(model_parameters, optimizer_name, lr=1e-3, weight_decay=0):
    optimizer_key = optimizer_name.lower()
    trainable_params = [p for p in model_parameters if p.requires_grad]
    if optimizer_key == 'radam':
        return RAdam(trainable_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_key == 'adadelta':
        return Adadelta(trainable_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_key == 'adam':
        return Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_key == 'adamw':
        return AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_key == 'nadam':
        return NAdam(trainable_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_key == 'sgd':
        return SGD(trainable_params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_key}.")


def build_checkpoint_name(args):
    filename = (f'repeat{args.repeat}_sub{args.sub}_bs{args.bs}_epochs{args.epochs}_lr{args.lr}_wd{args.wd}_'
                f'in_size{args.in_size}_out_size{args.out_size}_{args.metric}.pt')

    if hasattr(args, "test_session") and hasattr(args, "val_session"):
        filename = filename.replace(".pt", f'_testS{args.test_session}_valS{args.val_session}.pt')

    return filename


def save_results(results, results_path):
    results = results.detach().cpu().numpy()
    mean = np.nanmean(results).item()
    repeat_means = np.nanmean(results, axis=0)
    std = np.nanstd(repeat_means, ddof=0).item()
    subject_means = [f'{x:.4f}' for x in np.nanmean(results, axis=1).tolist()]
    subject_stds = [f'{x:.4f}' for x in np.nanstd(results, axis=1, ddof=0).tolist()]
    print(f"mean:{mean:.2f}\tstd:{std:.2f}")
    header_info = 'Mean: {:.4f}\t'.format(mean) + 'Std: {:.4f}\n'.format(std) \
                  + f'St.Mean: {", ".join(subject_means)}\n' + f'St.Std:  {", ".join(subject_stds)}\n'

    np.savetxt(results_path, results, fmt='%.4f', comments='', delimiter='\t', header=header_info)


def train_model(model, train_loader, valid_loader, test_loader, args):

    use_bce = args.nclass == 1
    loss_fn = nn.BCEWithLogitsLoss() if use_bce else nn.CrossEntropyLoss()

    optimizer = create_optimizer(model.parameters(), args.optim, lr=args.lr, weight_decay=args.wd)
    evaluator = Evaluator(test_loader)
    swd_weight = args.swd_weight
    swd_num_projections = args.swd_num_projections
    swd_p = args.swd_p
    swd_random_state = args.swd_random_state
    swd_metric = args.swd_metric
    swd_input_space = args.swd_input_space

    best_val_loss = 1e10
    best_score = 0
    for epoch in range(args.iterations):

        model.train()
        train_correct, train_samples = 0, 0
        val_correct, val_samples = 0, 0
        train_loss_sum, val_loss_sum = 0, 0
        train_swd_sum = 0.0
        for batch in train_loader:
            xb, yb, db = batch
            train_samples += yb.shape[0]
            features = model.forward_features(xb)
            logits = model.forward_head(features)
            yb = yb.to(logits.device)
            if use_bce:
                yb = yb.float()
            else:
                yb = yb.long()
            loss = loss_fn(logits, yb)
            if swd_weight > 0:
                features = features.mean(dim=1)
                db = db.to(features.device)
                swd_value = corsw.swd_loss(
                    features,
                    db,
                    metric=swd_metric,
                    input_space=swd_input_space,
                    num_projections=swd_num_projections,
                    p=swd_p,
                    random_state=swd_random_state,
                )
                loss = loss + swd_weight * swd_value
                train_swd_sum += swd_value.detach().item() * yb.shape[0]
            optimizer.zero_grad()

            if use_bce:
                pred_labels = torch.gt(torch.sigmoid(logits), 0.5).long()
                train_correct += (pred_labels == (yb > 0.5).long()).sum().item()
            else:
                pred_labels = logits.argmax(dim=1)
                train_correct += (pred_labels == yb).sum().item()

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * yb.shape[0]
        model.eval()

        with torch.no_grad():
            for batch in valid_loader:
                xb, yb, _ = batch
                val_samples += yb.shape[0]
                logits = model(xb)
                yb = yb.to(logits.device)
                if use_bce:
                    yb = yb.float()
                    pred_labels = torch.gt(torch.sigmoid(logits), 0.5).long()
                    val_correct += (pred_labels == (yb > 0.5).long()).sum().item()
                else:
                    yb = yb.long()
                    pred_labels = logits.argmax(dim=1)
                    val_correct += (pred_labels == yb).sum().item()

                val_loss_sum += loss_fn(logits, yb).item() * yb.shape[0]

        if val_loss_sum < best_val_loss:
            os.makedirs(args.model_path, exist_ok=True)
            best_val_loss = val_loss_sum
            checkpoint_name = build_checkpoint_name(args)
            checkpoint_path = os.path.join(args.model_path, checkpoint_name)
            torch.save(model.state_dict(), checkpoint_path)
            best_model = copy.deepcopy(model)
            state_dict = torch.load(checkpoint_path, map_location=next(model.parameters()).device)
            best_model.load_state_dict(state_dict)
            best_model.to(next(model.parameters()).device)
            if args.scoring_metric == 'acc':
                best_score = evaluator.get_accuracy(best_model)
            elif args.scoring_metric == 'auc':
                _, _, best_score, _ = evaluator.get_metrics_for_binaryclass(best_model)
        if epoch == 0 or epoch % 10 == 0 or epoch == args.iterations - 1:
            swd_text = ""
            if swd_weight > 0:
                swd_text = (
                    f' swd_loss[{swd_metric},{swd_input_space}]:'
                    f'{train_swd_sum / train_samples:.4f}'
                )
            print(
                f'epoch:{epoch + 1:03d}/{args.iterations} '
                f'train_loss:{train_loss_sum / train_samples:.4f} train_acc:{train_correct / train_samples:.4f} '
                f'val_loss:{val_loss_sum / val_samples:.4f} val_acc:{val_correct / val_samples:.4f} '
                f'test_{args.scoring_metric}:{best_score:.4f}{swd_text}')

    return best_score


class Evaluator:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def get_accuracy(self, model):
        model.eval()
        device = next(model.parameters()).device

        truths = []
        preds = []
        with torch.no_grad():
            for batch in tqdm(self.data_loader, mininterval=1):
                x, y = batch[0], batch[1]
                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                pred_y = logits.argmax(dim=-1)

                truths += y.cpu().squeeze().numpy().tolist()
                preds += pred_y.cpu().squeeze().numpy().tolist()

        truths = np.array(truths)
        preds = np.array(preds)
        return accuracy_score(truths, preds)

    def get_metrics_for_binaryclass(self, model):
        model.eval()
        device = next(model.parameters()).device

        truths = []
        preds = []
        scores = []
        with torch.no_grad():
            for batch in tqdm(self.data_loader, mininterval=1):
                x, y = batch[0], batch[1]
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                score_y = torch.sigmoid(pred).squeeze(-1)
                pred_y = torch.gt(score_y, 0.5).long()
                truths += y.long().cpu().squeeze().numpy().tolist()
                preds += pred_y.cpu().squeeze().numpy().tolist()
                scores += score_y.cpu().numpy().tolist()

        truths = np.array(truths)
        preds = np.array(preds)
        scores = np.array(scores)
        acc = balanced_accuracy_score(truths, preds)
        roc_auc = roc_auc_score(truths, scores)
        precision, recall, thresholds = precision_recall_curve(truths, scores, pos_label=1)
        pr_auc = auc(recall, precision)
        cm = confusion_matrix(truths, preds)
        return acc, pr_auc, roc_auc, cm
