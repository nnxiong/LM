# test_finetune.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class MNIST_DNN(nn.Module):
    def __init__(self):
        super(MNIST_DNN, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10)  # 10 classes
        )

    def forward(self, x):
        return self.net(x)

def train_model(model, train_loader, criterion, optimizer, epochs=20, report_every=5):
    loss_list, acc_list = [], []
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to("cpu"), labels.to("cpu")
            outputs = model(images)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = correct / total
        loss_list.append(total_loss)
        acc_list.append(acc)

        if epoch % report_every == 0:
            print(f"Epoch {epoch}: Loss={total_loss:.4f}, Accuracy={acc:.4f}")
    
    return loss_list, acc_list, model

# def run_finetune(params):
#     transform = transforms.Compose([transforms.ToTensor()])
#     train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
#     train_loader = DataLoader(train_dataset, batch_size=params.get("batch_size", 64), shuffle=True)

#     model = MNIST_DNN()
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=params.get("lr", 0.001))

#     loss_list, acc_list, model = train_model(
#         model, train_loader, criterion, optimizer, epochs=params.get("epochs", 20)
#     )

#     return {
#         "epochs": params.get("epochs", 20),
#         "losses": loss_list,
#         "accuracies": acc_list
#     }, model


def run_test_finetune(customized_input, param_dict, model_path="../model_path/test", output_model_path="../datasets/finetuned_model_path/test"):
    fintune_train_dataset = customized_input["train_dataset"]        # 网络path 或者 本机地址
    fintune_test_dataset = customized_input["test_dataset"]          # 网络path 或者 本机地址
    fintune_valid_dataset = customized_input.get("valid_dataset",[]) # 网络path 或者 本机地址

    finetune_mode = param_dict["finetune_mode"]  # full, lora, qlora, p_tuning, prefix_tuning ...
    config_param = param_dict["config_param"]    # e.g.    lora   {"r":8, "alpha":16}
    show_frequecy = param_dict["show_frequecy"]          # 每 show_frequecy 返回一次结果 1-10
    
    measurement = param_dict["measurement"]   # step or epoch 
    total_epoch_number = param_dict["total_epoch_number"]

    # 保存结果


    for epoch_idx in range(total_epoch_number):
        yield {
            "idx": show_frequecy times,                    # 必有字段    1-total_epoch_number

            "fig_param":{
                "x_axis": [1,2,...],
                "x_label": "step",
                "losses": loss_list,                # 必有字段
                "accuracies": acc_list,             # 可以有其他的指标
                "valid_losses":valid_loss_list,
                "valid_accuracies": valid_acc_list,
            },

            "test_accuracy": test_acc,              # 最后一轮输出前都是 None，最后一轮的时候顺便做test acc
            "output_model_path": output_path,        # 每show_frequecy保存一次当前最好结果对应的checkpoint
            "output_log_path": output_log_path,          # 
        }


从而可以包装整个finetune函数，从而可以在post “customized_input, param_dict, model_path="../model_path/test", output_model_path="../datasets/finetuned_model_path/test"” 之后
每个epoch_idx的时候返回当前的losses, accuracies, valid_losses, valid_accuracies等信息



