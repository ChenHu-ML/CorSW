import os
import torch
from torch.utils import data
from scipy import io


# session splits are configurable; defaults to train 123, val 4, test 5.
def get_all_dataloader(args):
    dev = args.device
    train = io.loadmat(os.path.join(args.data_path, 'U' + f'{args.sub:03d}' + '.mat'))
    temp_data = torch.Tensor(train['x_test'])
    temp_label = torch.Tensor(train['y_test']).view(-1)

    num_sessions = args.num_sessions
    total_samples = temp_data.shape[0]
    session_len = total_samples // num_sessions
    session_indices = []
    for session_id in range(num_sessions):
        start = session_id * session_len
        end = (session_id + 1) * session_len if session_id < num_sessions - 1 else total_samples
        session_indices.append(torch.arange(start, end))

    session_ids = torch.empty(total_samples, dtype=torch.long)
    for session_id, idx in enumerate(session_indices):
        session_ids[idx] = session_id

    test_session = args.test_session
    val_session = args.val_session

    test_idx = test_session - 1
    val_idx = val_session - 1
    train_sessions = [s for s in range(num_sessions) if s not in (test_idx, val_idx)]

    train_idx = torch.cat([session_indices[s] for s in train_sessions])
    valid_idx = session_indices[val_idx]
    test_idx = session_indices[test_idx]

    # 划分训练集，验证集，测试集
    x_train = temp_data[train_idx]
    y_train = temp_label[train_idx]
    x_valid = temp_data[valid_idx]
    y_valid = temp_label[valid_idx]
    x_test = temp_data[test_idx]
    y_test = temp_label[test_idx]
    d_train = session_ids[train_idx]
    d_valid = session_ids[valid_idx]
    d_test = session_ids[test_idx]

    x_train = x_train.to(dev)
    y_train = y_train.long().to(dev)
    x_valid = x_valid.to(dev)
    y_valid = y_valid.long().to(dev)
    x_test = x_test.to(dev)
    y_test = y_test.long().to(dev)
    d_train = d_train.long().to(dev)
    d_valid = d_valid.long().to(dev)
    d_test = d_test.long().to(dev)

    print(x_train.shape)
    print(y_train.shape)
    print(x_valid.shape)
    print(y_valid.shape)
    print(x_test.shape)
    print(y_test.shape)

    train_dataset = data.TensorDataset(x_train, y_train, d_train)
    valid_dataset = data.TensorDataset(x_valid, y_valid, d_valid)
    test_dataset = data.TensorDataset(x_test, y_test, d_test)

    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=0,
    )
    valid_loader = data.DataLoader(
        dataset=valid_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=0,
    )
    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, valid_loader, test_loader
