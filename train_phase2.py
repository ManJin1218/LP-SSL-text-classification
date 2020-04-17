import argparse
import pickle

import torch
import torch.nn as nn
import torch.optim as optim

from train_utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_epochs", type=int, help="total number of epochs to train with updating pseudo labels", required=True)
    parser.add_argument("--num_epochs", default=1, type=int, help="num epoch")
    parser.add_argument("--name", default="phase2", type=str, help="name of the phase2 model")
    parser.add_argument("--num_labeled", default=4250, type=int, help="number of labeled data used in make_data.py", required=True)
    parser.add_argument("--knn", default=100, type=int, help="k for knn")
    parser.add_argument("--phase1_model_name", default="baseline", type=str, help="name of the baseline/phase1 model", required=True)
    parser.add_argument("-t", "--model_type", type=str, help="type of tokenization", required=True, choices=["gru", "bert"])
    parser.add_argument("--hidden_dim", type=int, help="hidden dim", required=True)
    parser.add_argument("--num_layers", default=2, type=int, help="number of layers for GRU")
    parser.add_argument("--embedding_dim", default=300, type=int, help="embedding dim")
    parser.add_argument("--seed", default=1211, type=int, help="random seed")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # load dataloaders
    with open("data_local/processed/data_{}_{}.pickle".format(args.model_type, args.num_labeled), 'rb') as handle:
        d = pickle.load(handle)

    # path of the baseline/phase1 model
    PATH = "models/{}_model.pt".format(args.phase1_model_name)

    # config of model
    model_config = torch.load(PATH, map_location=torch.device(device))["args"]

    # epoch 0
    batch_features = extract_features(d["groundtruth_loader"], model_path=PATH, device=device)
    p_labels, updated_weights, updated_class_weights = label_propagation(batch_features, d["groundtruth_labels"], d["labeled_idx"], d["unlabeled_idx"], k=args.knn)
    pseudo_loader = update_pseudoloader(d["all_indices"], p_labels, updated_weights, updated_class_weights)
    model = create_model(model_config, phase2=True)
    model.load_state_dict(torch.load(PATH, map_location=torch.device(device))["model_state_dict"])
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.Adam(model.parameters())
    print("Epoch 0")
    loss, accuracy = train(pseudo_loader, d["val_loader"], model, optimizer, criterion, device, args)

    # path of the phase2 model
    PATH = "models/{}_{}_{}_model.pt".format(args.name, args.model_type, args.num_labeled)
    
    train_loss_history = []
    val_accuracy_history = []
    train_loss_history.append(loss)
    val_accuracy_history.append(accuracy)
    best_val_acc = 0
    early_stop = 5
    early_stop_cnt = 0

    # epoch 1-T'
    for i in range(args.total_epochs):
        print("Epoch {}".format(i + 1))
        batch_features = extract_features(d["groundtruth_loader"], model_path=PATH, device=device)
        p_labels, updated_weights, updated_class_weights = label_propagation(batch_features, d["groundtruth_labels"], d["labeled_idx"], d["unlabeled_idx"], k=args.knn)
        pseudo_loader = update_pseudoloader(d["all_indices"], p_labels, updated_weights, updated_class_weights)
        model = create_model(model_config, phase2=True)
        model.load_state_dict(torch.load(PATH, map_location=torch.device(device))["model_state_dict"])
        model = model.to(device)
        criterion = nn.CrossEntropyLoss(reduction="none")
        optimizer = optim.Adam(model.parameters())
        loss, accuracy = train(pseudo_loader, d["val_loader"], model, optimizer, criterion, device, args)
        train_loss_history.append(loss)
        val_accuracy_history.append(accuracy)
        if accuracy > best_val_acc:
            early_stop_cnt = 0
            best_val_acc = accuracy
            torch.save({
                "model_state_dict": model.state_dict(),
                "train_loss_history": train_loss_history,
                "val_accuracy_history": val_accuracy_history,
                "args": args
            }, "models/best_{}_{}_{}_model.pt".format(args.name, args.model_type, args.num_labeled))
        else:
            early_stop_cnt += 1
            if early_stop_cnt >= early_stop:
                return


if __name__ == "__main__":
    main()
