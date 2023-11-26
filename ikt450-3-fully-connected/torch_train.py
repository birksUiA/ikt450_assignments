import torch
import torch.nn.functional as F
import numpy as np
import high_level_network


def train(
    model: high_level_network.SimpleNet,
    optimizer: torch.optim.Optimizer,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 16,
):
    # Train
    print("Traning...")
    losses = []
    accuracy = []
    precision = []
    recall = []
    F1 = []
    val_loss = []
    val_accuracy = []
    val_precision = []
    val_recall = []
    val_F1 = []

    for e in range(1000):
        running_loss = 0
        TP, TN, FP, FN = 0, 0, 0, 0

        # suffle the dataset.
        shuffeled_indicers = np.random.permutation(len(x_train))
        x_train = x_train[shuffeled_indicers]
        y_train = y_train[shuffeled_indicers]

        model.train(True)
        for i in range(0, len(x_train), batch_size):
            optimizer.zero_grad()
            x = torch.tensor(x_train[i : i + batch_size], dtype=torch.float32)
            y = torch.tensor(y_train[i : i + batch_size], dtype=torch.float32)
            y_hat = model(x)
            y_hat = torch.squeeze(y_hat)

            loss = F.binary_cross_entropy(y, y_hat)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # cal TP, TN, FP, FN
            for y_hat_i, y_i, y_o in zip(y_hat, y, y_hat):
                if y_hat_i == 1 and y_i == 1:
                    TP += 1
                if y_hat_i == 0 and y_i == 0:
                    TN += 1
                if y_hat_i == 1 and y_i == 0:
                    FP += 1
                if y_hat_i == 0 and y_i == 1:
                    FN += 1

        # cal accuracy
        accuracy += [(TP + TN) / (TP + TN + FP + FN)]
        # cal precision
        precision += [(TP / (TP + FP)) if TP + FP != 0 else 0]
        # cal recall
        recall += [(TP / (TP + FN)) if TP + FN != 0 else 0]
        # cal F1
        F1 += [
            2 * precision[-1] * recall[-1] / (precision[-1] + recall[-1])
            if precision[-1] + recall[-1] != 0
            else 0
        ]
        # mean loss
        losses += [running_loss / len(x_train)]
        # validation
        (
            val_loss_i,
            val_accuracy_i,
            val_precision_i,
            val_recall_i,
            val_F1_i,
            TP_v,
            TN_v,
            FP_v,
            FN_v,
            m_v,
        ) = validat(model, x_val, y_val)
        val_loss += [val_loss_i]
        val_accuracy += [val_accuracy_i]
        val_precision += [val_precision_i]
        val_recall += [val_recall_i]
        val_F1 += [val_F1_i]

        if e % 20 == 0:
            # repport on a single line
            print(
                f"    Loss: {losses[-1]:<10.4f} |     Accuracy: {accuracy[-1]:<10.4f} |     Precision: {precision[-1]:<10.4f} |      Recall: {recall[-1]:<10.4f} |      F1: {F1[-1]:<10.4f}"
            )
            print(
                f"Val Loss: {val_loss[-1]:<10.4f} | Val Accuracy: {val_accuracy[-1]:<10.4f} | Val Precision: {val_precision[-1]:<10.4f} | Val Recall: {val_recall[-1]:<10.4f} | Val F1: {val_F1[-1]:<10.4f}"
            )
            print(
                f"TP:   {TP:3d} | TN:   {TN:3d} | FP:   {FP:3d} | FN:   {FN:3d}, model: {model._version}"
            )
            print(
                f"TP_v: {TP_v:3d} | TN_v: {TN_v:3d} | FP_v: {FP_v:3d} | FN_v: {FN_v:3d}, model: {m_v}"
            )

    return (
        losses,
        accuracy,
        precision,
        recall,
        F1,
        val_loss,
        val_accuracy,
        val_precision,
        val_recall,
        val_F1,
    )


def validat(model: high_level_network.SimpleNet, x_val, y_val):
    # Test
    TP, TN, FP, FN = 0, 0, 0, 0
    loss = 0
    model.eval()
    for x, y in zip(x_val, y_val):
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        # make x 2d
        x = x.unsqueeze(0)
        y_hat = model(x)
        y_hat = torch.squeeze(y_hat)

        if y_hat == 1 and y == 1:
            TP += 1
        if y_hat == 0 and y == 0:
            TN += 1
        if y_hat == 1 and y == 0:
            FP += 1
        if y_hat == 0 and y == 1:
            FN += 1
        loss += F.binary_cross_entropy(y, y_hat)

    # cal accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    # cal precision
    precision = (TP / (TP + FP)) if TP + FP != 0 else 0
    # cal recall
    recall = (TP / (TP + FN)) if TP + FN != 0 else 0
    # cal F1
    F1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

    return (
        loss / len(x_val),
        accuracy,
        precision,
        recall,
        F1,
        TP,
        TN,
        FP,
        FN,
        model._version,
    )


def test(model, x_test, y_test):
    loss, accuracy, precision, recall, F1, TP, TN, FP, FN, model = validat(
        model, x_test, y_test
    )

    print(model)
    print("BCE loss: ", loss)
    print("accuracy: ", accuracy)
    print("precision: ", precision)
    print("recall: ", recall)
    print("F1: ", F1)


def plot_loss(losses):
    # plot running loss over epochs
    import matplotlib.pyplot as plt

    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss over epochs")
    plt.savefig("ass2-loss-over-epoch.png", dpi=300)
    plt.show()
