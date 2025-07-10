
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from datetime import datetime
from torchvision import datasets, transforms
from torch.utils.data import random_split


class test_DNN(nn.Module):
    def __init__(self):
        super(test_DNN, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.net(x)

def evaluate(model, loader, device="cpu"):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            preds.extend(pred.tolist())
            labels.extend(y.tolist())
    return accuracy_score(labels, preds)

def log_line(text, path):
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(f"{time_str} INFO {text}\n")

def run_test_finetune(customized_input, param_dict, model_path="../model_path/test",
                      output_model_path="../finetuned_model_path/test/task_1_0704",
                      output_log_path="../finetuned_model_path/test/finetune.log"):

    train_dataset_path = customized_input["train_dataset_path"]
    test_dataset_path = customized_input["test_dataset_path"]
    if "valid_dataset_path" not in customized_input:
        raise ValueError("You must provide 'valid_dataset'.")
    valid_dataset_path = customized_input["valid_dataset_path"]

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root="/app/finetune_datasets/test/train", train=True, download=False, transform=transform)
    all_test_dataset = datasets.MNIST(root="/app/finetune_datasets/test/test", train=False, download=False, transform=transform)
    valid_dataset, test_dataset = random_split(all_test_dataset, [5000, 5000])

    show_frequency = param_dict.get("show_frequecy", 5)
    measurement = param_dict.get("measurement", "epoch")
    total_epoch_number = param_dict.get("total_epoch_number", 20)
    batch_size = param_dict.get("batch_size", 64)
    lr = param_dict.get("lr", 1e-3)

    if measurement not in ["epoch", "step"]:
        raise ValueError(f"Invalid measurement '{measurement}'. Supported values: 'epoch' or 'step'.")

    os.makedirs(output_model_path, exist_ok=True)
    log_line("Starting finetuning...", output_log_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = test_DNN().to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_valid_acc = 0.0
    best_model_path = None

    loss_list, acc_list = [], []
    valid_loss_list, valid_acc_list = [], []
    x_axis_list = []

    step_count = 0
    is_final_flag = False

    for epoch in range(1, total_epoch_number + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            step_count += 1

            if measurement == "step" and step_count % show_frequency == 0:
                epoch_loss = total_loss
                epoch_acc = correct / total
                x_axis_list.append(step_count)
                loss_list.append(epoch_loss)
                acc_list.append(epoch_acc)

                model.eval()
                val_loss, val_correct, val_total = 0, 0, 0
                with torch.no_grad():
                    for x_val, y_val in valid_loader:
                        x_val, y_val = x_val.to(device), y_val.to(device)
                        out = model(x_val)
                        val_loss += criterion(out, y_val).item()
                        pred = out.argmax(dim=1)
                        val_correct += (pred == y_val).sum().item()
                        val_total += y_val.size(0)
                valid_acc = val_correct / val_total
                valid_loss_list.append(val_loss)
                valid_acc_list.append(valid_acc)

                is_best = False
                if valid_acc >= best_valid_acc:
                    best_valid_acc = valid_acc
                    is_best = True
                    if best_model_path and os.path.exists(best_model_path):
                        os.remove(best_model_path)
                    best_model_path = os.path.join(output_model_path, f"best_model_step{step_count}.pth")
                    torch.save(model.state_dict(), best_model_path)
                    log_line(f"Saved best model at step {step_count} (acc={valid_acc:.4f}).", output_log_path)

                log_line(f"Step {step_count}: loss={epoch_loss:.3f}, acc={epoch_acc:.4f}", output_log_path)

                yield {
                    "is_final": False,
                    "idx": step_count,
                    "fig_param": {
                        "x_axis": x_axis_list,
                        "x_label": "step",
                        "losses": loss_list[:],
                        "accuracies": acc_list[:],
                        "valid_losses": valid_loss_list,
                        "valid_accuracies": valid_acc_list,
                    },
                    "test_accuracy": None,
                    "output_model_path": output_model_path,
                    "output_log_path": output_log_path,
                }

        if measurement == "epoch" and epoch % show_frequency == 0:
            epoch_loss = total_loss
            epoch_acc = correct / total
            x_axis_list.append(epoch)
            loss_list.append(epoch_loss)
            acc_list.append(epoch_acc)

            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            with torch.no_grad():
                for x_val, y_val in valid_loader:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    out = model(x_val)
                    val_loss += criterion(out, y_val).item()
                    pred = out.argmax(dim=1)
                    val_correct += (pred == y_val).sum().item()
                    val_total += y_val.size(0)
            valid_acc = val_correct / val_total
            valid_loss_list.append(val_loss)
            valid_acc_list.append(valid_acc)

            is_best = False
            if valid_acc >= best_valid_acc:
                best_valid_acc = valid_acc
                is_best = True
                if best_model_path and os.path.exists(best_model_path):
                    os.remove(best_model_path)
                best_model_path = os.path.join(output_model_path, f"best_model_epoch{epoch}.pth")
                torch.save(model.state_dict(), best_model_path)
                log_line(f"Saved best model at epoch {epoch} (acc={valid_acc:.4f})", output_log_path)

            log_line(f"Epoch {epoch}: loss={epoch_loss:.3f}, acc={epoch_acc:.4f}", output_log_path)

            yield {
                "is_final": False,
                "idx": epoch,
                "fig_param": {
                    "x_axis": x_axis_list,
                    "x_label": "epoch",
                    "losses": loss_list[:],
                    "accuracies": acc_list[:],
                    "valid_losses": valid_loss_list,
                    "valid_accuracies": valid_acc_list,
                },
                "test_accuracy": None,
                "output_model_path": output_model_path,
                "output_log_path": output_log_path,
            }

    is_final_flag = True
    test_acc = evaluate(model, test_loader, device)
    log_line(f"Final Test Accuracy: {test_acc:.4f}", output_log_path)

    if measurement == "step" and (step_count % show_frequency != 0):
        x_axis_list.append(step_count)
        epoch_loss = total_loss
        epoch_acc = correct / total
        loss_list.append(epoch_loss)
        acc_list.append(epoch_acc)

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for x_val, y_val in valid_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                out = model(x_val)
                val_loss += criterion(out, y_val).item()
                pred = out.argmax(dim=1)
                val_correct += (pred == y_val).sum().item()
                val_total += y_val.size(0)
        valid_acc = val_correct / val_total
        valid_loss_list.append(val_loss)
        valid_acc_list.append(valid_acc)

        if valid_acc >= best_valid_acc:
            best_valid_acc = valid_acc
            best_model_path = os.path.join(output_model_path, f"best_model_step{step_count}.pth")
            torch.save(model.state_dict(), best_model_path)
            log_line(f"Saved best model at step {step_count} (acc={valid_acc:.4f}).", output_log_path)

    if measurement == "epoch" and (total_epoch_number % show_frequency != 0):
        x_axis_list.append(total_epoch_number)
        epoch_loss = total_loss
        epoch_acc = correct / total
        loss_list.append(epoch_loss)
        acc_list.append(epoch_acc)

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for x_val, y_val in valid_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                out = model(x_val)
                val_loss += criterion(out, y_val).item()
                pred = out.argmax(dim=1)
                val_correct += (pred == y_val).sum().item()
                val_total += y_val.size(0)
        valid_acc = val_correct / val_total
        valid_loss_list.append(val_loss)
        valid_acc_list.append(valid_acc)

        if valid_acc >= best_valid_acc:
            best_valid_acc = valid_acc
            best_model_path = os.path.join(output_model_path, f"best_model_epoch{total_epoch_number}.pth")
            torch.save(model.state_dict(), best_model_path)
            log_line(f"Saved best model at epoch {total_epoch_number} (acc={valid_acc:.4f}).", output_log_path)

    yield {
        "is_final": True,
        "idx": step_count if measurement == "step" else total_epoch_number,
        "fig_param": {
            "x_axis": x_axis_list,
            "x_label": measurement,
            "losses": loss_list[:],
            "accuracies": acc_list[:],
            "valid_losses": valid_loss_list,
            "valid_accuracies": valid_acc_list,
        },
        "test_accuracy": test_acc,
        "output_model_path": output_model_path,
        "output_log_path": output_log_path,
    }



if __name__ == "__main__":

    # 参数
    param_dict = {
        "finetune_mode": "full",
        "config_param": {},
        "show_frequecy": 2,
        "measurement": "epoch",  # or "step"
        "total_epoch_number": 5,
        "batch_size": 64,
        "lr": 0.001
    }

    # 输入字典
    customized_input = {
        "train_dataset_path": "/app/finetune_datasets/test/train",
        "valid_dataset_path": "/app/finetune_datasets/test/test",   # "/app/finetune_datasets/test/valid"  # 这里使用了测试集拆成了valid和train， 大家写的时候正常写valid
        "test_dataset_path": "/app/finetune_datasets/test/test"
    }

    for output in run_test_finetune(customized_input, param_dict, model_path="../model_path/test",
                      output_model_path="../finetuned_model_path/test/20250423020919_task_2_1/model_files",   # /mnt/data1/bioford2/finetuned_model_path/test/task_1_0704/best_model_epoch5.pth
                      output_log_path="../finetuned_model_path/test/20250423020919_task_2_1/finetune.log" ):
        print("FT_RESULT:", output)


# FT_RESULT: {'is_final': False, 'idx': 2, 'fig_param': {'x_axis': [2], 'x_label': 'epoch', 'losses': [113.17636464256793], 'accuracies': [0.96325], 'valid_losses': [8.439233392477036], 'valid_accuracies': [0.966]}, 'test_accuracy': None, 'output_model_path': '../finetuned_model_path/test/20250423020919_task_2_1/model_files', 'output_log_path': '../finetuned_model_path/test/20250423020919_task_2_1/finetune.log'}
# FT_RESULT: {'is_final': False, 'idx': 4, 'fig_param': {'x_axis': [2, 4], 'x_label': 'epoch', 'losses': [113.17636464256793, 55.28447434864938], 'accuracies': [0.96325, 0.9819], 'valid_losses': [8.439233392477036, 6.849441746599041], 'valid_accuracies': [0.966, 0.973]}, 'test_accuracy': None, 'output_model_path': '../finetuned_model_path/test/20250423020919_task_2_1/model_files', 'output_log_path': '../finetuned_model_path/test/20250423020919_task_2_1/finetune.log'}
# FT_RESULT: {'is_final': True, 'idx': 5, 'fig_param': {'x_axis': [2, 4, 5], 'x_label': 'epoch', 'losses': [113.17636464256793, 55.28447434864938, 42.89073523168918], 'accuracies': [0.96325, 0.9819, 0.9857166666666667], 'valid_losses': [8.439233392477036, 6.849441746599041, 7.862476927693933], 'valid_accuracies': [0.966, 0.973, 0.9722]}, 'test_accuracy': 0.9766, 'output_model_path': '../finetuned_model_path/test/20250423020919_task_2_1/model_files', 'output_log_path': '../finetuned_model_path/test/20250423020919_task_2_1/finetune.log'}



# use step as measure unit
    param_dict = {
        "finetune_mode": "full",
        "config_param": {},
        "show_frequecy": 200,
        "measurement": "step",  # or "step"
        "total_epoch_number": 2,
        "batch_size": 64,
        "lr": 0.001
    }

    # 输入字典
    customized_input = {
        "train_dataset_path": "/app/finetune_datasets/test/train",
        "valid_dataset_path": "/app/finetune_datasets/test/test",   # "/app/finetune_datasets/test/valid"  # 这里使用了测试集拆成了valid和train， 大家写的时候正常写valid
        "test_dataset_path": "/app/finetune_datasets/test/test"
    }

    for output in run_test_finetune(customized_input, param_dict, model_path="../model_path/test",
                    output_model_path="../finetuned_model_path/test/20250423020919_task_2_1/model_files",   # /mnt/data1/bioford2/finetuned_model_path/test/task_1_0704/best_model_epoch5.pth
                    output_log_path="../finetuned_model_path/test/20250423020919_task_2_1/finetune.log" ):
        print("FT_RESULT:", output)


# FT_RESULT: {'is_final': False, 'idx': 200, 'fig_param': {'x_axis': [200], 'x_label': 'step', 'losses': [126.46783593297005], 'accuracies': [0.82359375], 'valid_losses': [23.863651856780052], 'valid_accuracies': [0.9142]}, 'test_accuracy': None, 'output_model_path': '../finetuned_model_path/test/20250423020919_task_2_1/model_files', 'output_log_path': '../finetuned_model_path/test/20250423020919_task_2_1/finetune.log'}
# FT_RESULT: {'is_final': False, 'idx': 400, 'fig_param': {'x_axis': [200, 400], 'x_label': 'step', 'losses': [126.46783593297005, 181.04202397167683], 'accuracies': [0.82359375, 0.872421875], 'valid_losses': [23.863651856780052, 19.196646701544523], 'valid_accuracies': [0.9142, 0.9322]}, 'test_accuracy': None, 'output_model_path': '../finetuned_model_path/test/20250423020919_task_2_1/model_files', 'output_log_path': '../finetuned_model_path/test/20250423020919_task_2_1/finetune.log'}
# FT_RESULT: {'is_final': False, 'idx': 600, 'fig_param': {'x_axis': [200, 400, 600], 'x_label': 'step', 'losses': [126.46783593297005, 181.04202397167683, 224.60504310950637], 'accuracies': [0.82359375, 0.872421875, 0.8935677083333333], 'valid_losses': [23.863651856780052, 19.196646701544523, 14.229090549051762], 'valid_accuracies': [0.9142, 0.9322, 0.947]}, 'test_accuracy': None, 'output_model_path': '../finetuned_model_path/test/20250423020919_task_2_1/model_files', 'output_log_path': '../finetuned_model_path/test/20250423020919_task_2_1/finetune.log'}
# FT_RESULT: {'is_final': False, 'idx': 800, 'fig_param': {'x_axis': [200, 400, 600, 800], 'x_label': 'step', 'losses': [126.46783593297005, 181.04202397167683, 224.60504310950637, 261.8677057363093], 'accuracies': [0.82359375, 0.872421875, 0.8935677083333333, 0.90650390625], 'valid_losses': [23.863651856780052, 19.196646701544523, 14.229090549051762, 11.538524094969034], 'valid_accuracies': [0.9142, 0.9322, 0.947, 0.9552]}, 'test_accuracy': None, 'output_model_path': '../finetuned_model_path/test/20250423020919_task_2_1/model_files', 'output_log_path': '../finetuned_model_path/test/20250423020919_task_2_1/finetune.log'}
# FT_RESULT: {'is_final': False, 'idx': 1000, 'fig_param': {'x_axis': [200, 400, 600, 800, 1000], 'x_label': 'step', 'losses': [126.46783593297005, 181.04202397167683, 224.60504310950637, 261.8677057363093, 8.173539981245995], 'accuracies': [0.82359375, 0.872421875, 0.8935677083333333, 0.90650390625, 0.9591733870967742], 'valid_losses': [23.863651856780052, 19.196646701544523, 14.229090549051762, 11.538524094969034, 10.882396001368761], 'valid_accuracies': [0.9142, 0.9322, 0.947, 0.9552, 0.956]}, 'test_accuracy': None, 'output_model_path': '../finetuned_model_path/test/20250423020919_task_2_1/model_files', 'output_log_path': '../finetuned_model_path/test/20250423020919_task_2_1/finetune.log'}
# FT_RESULT: {'is_final': False, 'idx': 1200, 'fig_param': {'x_axis': [200, 400, 600, 800, 1000, 1200], 'x_label': 'step', 'losses': [126.46783593297005, 181.04202397167683, 224.60504310950637, 261.8677057363093, 8.173539981245995, 34.955071130767465], 'accuracies': [0.82359375, 0.872421875, 0.8935677083333333, 0.90650390625, 0.9591733870967742, 0.9601622137404581], 'valid_losses': [23.863651856780052, 19.196646701544523, 14.229090549051762, 11.538524094969034, 10.882396001368761, 9.399585172533989], 'valid_accuracies': [0.9142, 0.9322, 0.947, 0.9552, 0.956, 0.9656]}, 'test_accuracy': None, 'output_model_path': '../finetuned_model_path/test/20250423020919_task_2_1/model_files', 'output_log_path': '../finetuned_model_path/test/20250423020919_task_2_1/finetune.log'}
# FT_RESULT: {'is_final': False, 'idx': 1400, 'fig_param': {'x_axis': [200, 400, 600, 800, 1000, 1200, 1400], 'x_label': 'step', 'losses': [126.46783593297005, 181.04202397167683, 224.60504310950637, 261.8677057363093, 8.173539981245995, 34.955071130767465, 58.904851188883185], 'accuracies': [0.82359375, 0.872421875, 0.8935677083333333, 0.90650390625, 0.9591733870967742, 0.9601622137404581, 0.9620197510822511], 'valid_losses': [23.863651856780052, 19.196646701544523, 14.229090549051762, 11.538524094969034, 10.882396001368761, 9.399585172533989, 9.067163713276386], 'valid_accuracies': [0.9142, 0.9322, 0.947, 0.9552, 0.956, 0.9656, 0.966]}, 'test_accuracy': None, 'output_model_path': '../finetuned_model_path/test/20250423020919_task_2_1/model_files', 'output_log_path': '../finetuned_model_path/test/20250423020919_task_2_1/finetune.log'}
# FT_RESULT: {'is_final': False, 'idx': 1600, 'fig_param': {'x_axis': [200, 400, 600, 800, 1000, 1200, 1400, 1600], 'x_label': 'step', 'losses': [126.46783593297005, 181.04202397167683, 224.60504310950637, 261.8677057363093, 8.173539981245995, 34.955071130767465, 58.904851188883185, 81.02635871618986], 'accuracies': [0.82359375, 0.872421875, 0.8935677083333333, 0.90650390625, 0.9591733870967742, 0.9601622137404581, 0.9620197510822511, 0.9635337990936556], 'valid_losses': [23.863651856780052, 19.196646701544523, 14.229090549051762, 11.538524094969034, 10.882396001368761, 9.399585172533989, 9.067163713276386, 7.944749088026583], 'valid_accuracies': [0.9142, 0.9322, 0.947, 0.9552, 0.956, 0.9656, 0.966, 0.97]}, 'test_accuracy': None, 'output_model_path': '../finetuned_model_path/test/20250423020919_task_2_1/model_files', 'output_log_path': '../finetuned_model_path/test/20250423020919_task_2_1/finetune.log'}
# FT_RESULT: {'is_final': False, 'idx': 1800, 'fig_param': {'x_axis': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800], 'x_label': 'step', 'losses': [126.46783593297005, 181.04202397167683, 224.60504310950637, 261.8677057363093, 8.173539981245995, 34.955071130767465, 58.904851188883185, 81.02635871618986, 103.55943891964853], 'accuracies': [0.82359375, 0.872421875, 0.8935677083333333, 0.90650390625, 0.9591733870967742, 0.9601622137404581, 0.9620197510822511, 0.9635337990936556, 0.9644540313225058], 'valid_losses': [23.863651856780052, 19.196646701544523, 14.229090549051762, 11.538524094969034, 10.882396001368761, 9.399585172533989, 9.067163713276386, 7.944749088026583, 8.5378526346758], 'valid_accuracies': [0.9142, 0.9322, 0.947, 0.9552, 0.956, 0.9656, 0.966, 0.97, 0.969]}, 'test_accuracy': None, 'output_model_path': '../finetuned_model_path/test/20250423020919_task_2_1/model_files', 'output_log_path': '../finetuned_model_path/test/20250423020919_task_2_1/finetune.log'}
# FT_RESULT: {'is_final': True, 'idx': 1876, 'fig_param': {'x_axis': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 1876], 'x_label': 'step', 'losses': [126.46783593297005, 181.04202397167683, 224.60504310950637, 261.8677057363093, 8.173539981245995, 34.955071130767465, 58.904851188883185, 81.02635871618986, 103.55943891964853, 112.10872308537364], 'accuracies': [0.82359375, 0.872421875, 0.8935677083333333, 0.90650390625, 0.9591733870967742, 0.9601622137404581, 0.9620197510822511, 0.9635337990936556, 0.9644540313225058, 0.9647666666666667], 'valid_losses': [23.863651856780052, 19.196646701544523, 14.229090549051762, 11.538524094969034, 10.882396001368761, 9.399585172533989, 9.067163713276386, 7.944749088026583, 8.5378526346758, 7.080890180077404], 'valid_accuracies': [0.9142, 0.9322, 0.947, 0.9552, 0.956, 0.9656, 0.966, 0.97, 0.969, 0.9714]}, 'test_accuracy': 0.968, 'output_model_path': '../finetuned_model_path/test/20250423020919_task_2_1/model_files', 'output_log_path': '../finetuned_model_path/test/20250423020919_task_2_1/finetune.log'}
        


        