import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import time # 时间库，用于计算训练耗时

# 1.配置与超参数
# (1) 使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# (2) 对抗攻击参数
EPSILON = 8.0 / 255.0       # 最大扰动幅度
PGD_ALPHA = 2.0 / 255.0     # 单步扰动步长
PGD_ITERS = 10              # 攻击迭代次数

# (3) 训练参数
EPOCH = 30                  # ResNET-20模型的训练轮数
SURROGATE_MODEL_EPOCHS = 5  # 代理模型训练轮数
LEARNING_RATE = 0.001
BATCH_SIZE = 256

# (4) 目录和文件路径
DATA_DIR = './data'
TARGET_MODEL_PATH = os.path.join(DATA_DIR, 'resnet20.pth')
SURROGATE_MODEL_PATH = os.path.join(DATA_DIR, 'surrogate_model_cifar10.pth')
os.makedirs(DATA_DIR, exist_ok=True)

# 2.模型定义: 标准的 ResNet-20
class BasicBlock(nn.Module):
    # 残差定义：两个3X3卷积和残差连接
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 残差连接，处理维度或步长不匹配的情况
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    # ResNet-20架构：含3个残差层
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1) # Layer1:16通道，3个残差块
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2) # Layer2:32通道，步长2下采样
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2) #Layer3:64通道，步长2下采样
        self.fc = nn.Linear(64 * block.expansion, num_classes) 
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)  # 全局平均池化
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])

# 代理模型：简单的CNN
class SurrogateCNN(nn.Module):
    def __init__(self):
        super(SurrogateCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 3.数据加载与预处理
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2470, 0.2435, 0.2616)

# (1) 用于攻击的转换，得到 [0,1] 范围的 Tensor
transform_attack = transforms.Compose([transforms.ToTensor()])

# (2) 用于模型训练和评估的转换，包含归一化
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),   # 随机裁剪，增强泛化
    transforms.RandomHorizontalFlip(),      # 随机水平翻转
    transforms.ToTensor(), 
    transforms.Normalize(cifar10_mean, cifar10_std)
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)
])

# (3) 数据集加载
# 攻击用的测试集 (不归一化)
test_dataset_attack = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform_attack)
test_loader_attack = DataLoader(test_dataset_attack, batch_size=1, shuffle=False)

# 训练和评估用的数据集 (归一化)
train_dataset_train = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform_train)
train_loader_train = DataLoader(train_dataset_train, batch_size=BATCH_SIZE, shuffle=True)


# 4.模型训练函数
def train_model(model, train_loader, epochs, lr, model_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    end_time = time.time()
    print(f"Training Time Use: {end_time - start_time:.2f} seconds.")
    print(f"Saving model to {model_path}")
    torch.save(model.state_dict(), model_path)


# 5.对抗攻击算法实现: PGD算法
def pgd_attack(model, images, labels, eps, alpha, iters):
    images = images.clone().detach().to(device) # 复制原始图像（避免修改原值）
    labels = labels.clone().detach().to(device) # 保存原始图像用于扰动限制
    original_images = images.clone().detach()

    for i in range(iters):
        images.requires_grad = True
        # 因为模型输入要求，攻击时，需要对输入进行归一化
        images_normalized = transforms.Normalize(cifar10_mean, cifar10_std)(images)
        outputs = model(images_normalized)
        
        model.zero_grad()
        loss = F.cross_entropy(outputs, labels) # 分类损失
        loss.backward()

        # 沿使损失增加的方向，梯度符号法更新扰动
        adv_images = images + alpha * images.grad.sign()
        # 相对于原始图像，限制扰动在[-eps, eps]范围内
        eta = torch.clamp(adv_images - original_images, min=-eps, max=eps)
        # 确保对抗样本在合法像素范围[0,1]内
        images = torch.clamp(original_images + eta, min=0, max=1).detach()
    # 返回对抗样本
    return images


# 6.评估函数
def evaluate_attack(model, loader, attack_fn=None, attack_kwargs=None, surrogate_model=None):
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    model.eval()
    if surrogate_model:
        surrogate_model.eval()

    # loader 传入的是 test_loader_attack，其数据在 [0,1] 范围
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        # 生成对抗样本
        if attack_fn:
            # 黑盒攻击：使用代理模型生成对抗样本
            attack_model = surrogate_model if surrogate_model else model
            adv_images = attack_fn(attack_model, images, labels, **attack_kwargs)
            # 对抗样本需要归一化后才能输入模型
            input_images = transforms.Normalize(cifar10_mean, cifar10_std)(adv_images)
        else:
            # 干净样本，直接归一化
            input_images = transforms.Normalize(cifar10_mean, cifar10_std)(images)

        with torch.no_grad():
            outputs = model(input_images)
        _, predicted = torch.max(outputs.data, 1)# 获取预测类别
        
        total += labels.size(0)
        c = (predicted == labels).squeeze()
        correct += c.sum().item()

        # 计算每个类的准确率
        for i in range(labels.size(0)):
            label = labels[i]
            # 确保 c 是可迭代的
            current_c = c.item() if c.dim() == 0 else c[i].item()
            class_correct[label] += current_c
            class_total[label] += 1
    
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    
    for i in range(10):
        if class_total[i] > 0:
            print(f'Accuracy of class {i} : {100 * class_correct[i] / class_total[i]:.2f}%')
    
    return accuracy


# 7.主执行流程
def main():
    # (1)准备目标模型(ResNet-20)
    print("="*50)
    print("Preparing Target Model (ResNet-20)...")
    target_model = resnet20().to(device)

    if os.path.exists(TARGET_MODEL_PATH):
        print(f"Loading previously trained target model from {TARGET_MODEL_PATH}")
        target_model.load_state_dict(torch.load(TARGET_MODEL_PATH, map_location=device))
    else:
        print("No pre-trained target model found. Starting training...")
        train_model(target_model, train_loader_train, EPOCH, LEARNING_RATE, TARGET_MODEL_PATH)

    # (2)准备代理模型
    print("\n" + "="*50)
    print("Preparing Surrogate Model (Custom CNN)...")
    surrogate_model = SurrogateCNN().to(device)

    if os.path.exists(SURROGATE_MODEL_PATH):
        print(f"Loading previously trained surrogate model from {SURROGATE_MODEL_PATH}")
        surrogate_model.load_state_dict(torch.load(SURROGATE_MODEL_PATH, map_location=device))
    else:
        print("No pre-trained surrogate model found. Starting training...")
        train_model(surrogate_model, train_loader_train, SURROGATE_MODEL_EPOCHS, LEARNING_RATE, SURROGATE_MODEL_PATH)


    # (3)评估
    print("\n" + "="*50)
    print("1. Evaluating on CLEAN data (Original Accuracy)")

    evaluate_attack(target_model, test_loader_attack)
    
    print("\n" + "="*50)
    print(f"2. Evaluating WHITE-BOX PGD Attack (eps={EPSILON})")
    pgd_params = {'eps': EPSILON, 'alpha': PGD_ALPHA, 'iters': PGD_ITERS}
    evaluate_attack(target_model, test_loader_attack, attack_fn=pgd_attack, attack_kwargs=pgd_params)

    print("\n" + "="*50)
    print(f"3. Evaluating BLACK-BOX Transfer Attack (eps={EPSILON})")
    print("Attack generated on SurrogateCNN, tested on ResNet-20")
    evaluate_attack(
        model=target_model, 
        loader=test_loader_attack, 
        attack_fn=pgd_attack, 
        attack_kwargs=pgd_params, 
        surrogate_model=surrogate_model
    )

if __name__ == '__main__':
    main()