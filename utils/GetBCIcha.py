import os
import torch
from torch.utils import data
from scipy import io


# session 123 is training set, session4 is validation set, and session5 is testing set.
def get_all_dataloader(args):
    dev = args.device
    train = io.loadmat(os.path.join(args.data_path, f'Data_S{args.sub:02d}_Sess' + '.mat'))

    tempdata = torch.Tensor(train['x_test']).unsqueeze(1)
    templabel = torch.Tensor(train['y_test']).view(-1)
    total_samples = tempdata.shape[0]
    num_sessions = args.num_sessions
    if num_sessions == 5 and total_samples == 340:
        session_sizes = [60, 60, 60, 60, 100]
    else:
        base = total_samples // num_sessions
        session_sizes = [base] * (num_sessions - 1)
        session_sizes.append(total_samples - base * (num_sessions - 1))

    session_indices = []
    start = 0
    for size in session_sizes:
        end = start + size
        session_indices.append(torch.arange(start, end))
        start = end

    session_ids = torch.empty(total_samples, dtype=torch.long)
    for session_id, idx in enumerate(session_indices):
        session_ids[idx] = session_id

    test_session = args.test_session
    val_session = args.val_session
    if val_session <= 0:
        val_session = ((test_session - 2) % num_sessions) + 1

    test_idx = test_session - 1
    val_idx = val_session - 1
    train_sessions = [s for s in range(num_sessions) if s not in (test_idx, val_idx)]

    train_idx = torch.cat([session_indices[s] for s in train_sessions])
    valid_idx = session_indices[val_idx]
    test_idx = session_indices[test_idx]

    x_train = tempdata[train_idx]
    y_train = templabel[train_idx]
    x_valid = tempdata[valid_idx]
    y_valid = templabel[valid_idx]
    x_test = tempdata[test_idx]
    y_test = templabel[test_idx]
    d_train = session_ids[train_idx]
    d_valid = session_ids[valid_idx]
    d_test = session_ids[test_idx]

    x_train = x_train.to(dev)
    y_train = y_train.unsqueeze(1).float().to(dev)
    x_valid = x_valid.to(dev)
    y_valid = y_valid.unsqueeze(1).float().to(dev)
    x_test = x_test.to(dev)
    y_test = y_test.unsqueeze(1).float().to(dev)
    d_train = d_train.to(dev)
    d_valid = d_valid.to(dev)
    d_test = d_test.to(dev)

    print(x_train.shape)
    print(y_train.shape)
    print(x_valid.shape)
    print(y_valid.shape)
    print(x_test.shape)
    print(y_test.shape)

    train_dataset = data.TensorDataset(x_train, y_train, d_train)
    valid_dataset = data.TensorDataset(x_valid, y_valid, d_valid)
    test_dataset = data.TensorDataset(x_test, y_test, d_test)

    trainloader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=0,
    )
    validloader = data.DataLoader(
        dataset=valid_dataset,
        batch_size=60,
        shuffle=False,
        num_workers=0,
    )
    testloader = data.DataLoader(
        dataset=test_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=0,
    )

    return trainloader, validloader, testloader
