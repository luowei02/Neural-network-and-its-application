import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification,  get_scheduler
from torch.optim import AdamW
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm # For progress bars
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# 0. 库版本、CUDA检查
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available : {torch.cuda.is_available()}")
print(f"CUDA Version   : {torch.version.cuda}")

# 1. 定义参数和DEVICE
MODEL_NAME = "bert-base-uncased" # 使用Hugging Face上的预训练模型BERT
MAX_LENGTH = 256                 # 定义分词器的最大序列长度
BATCH_SIZE = 16                  # 训练和验证的批次大小
LEARNING_RATE = 2e-5             # AdamW优化器的学习率
NUM_EPOCHS = 3                   # 训练轮数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

OUTPUT_DIR = "./model_for_imdb" # 保存训练好的模型的路径

# 2. 数据集加载和预处理
print("\nLoading IMDB dataset and tokenizer: ")
raw_datasets = load_dataset("imdb")

print(f"Dataset loaded. Structure: {raw_datasets}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
    # padding='max_length': 把短的序列扩展到MAX_LENTH
    # truncation=True: 把长的序列修剪到MAX_LENTH

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# 移除"text"这栏，把数据转化为PyTorch张量
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["test"]

# 创建DataLoader，按照定义的BATCH_SIZE分割
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE)

# 3. 加载模型、定义优化器和学习率调度器等
print(f"\nLoading pre-trained Transformer model: {MODEL_NAME}")

# 标签分类为positive (1) and negative (0)，所以设置num_labels=2
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(DEVICE)
print("Model loaded and moved to device.")

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

num_training_steps = NUM_EPOCHS * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", # 对transformer的常规配置
    optimizer=optimizer,
    num_warmup_steps=0, # (0,0.1)
    num_training_steps=num_training_steps
)
print("Optimizer and scheduler seted.")

# 4. 模型训练
print("\nTraining start:")
progress_bar_train = tqdm(range(num_training_steps), desc="Training Progress")
progress_bar_eval = tqdm(range(NUM_EPOCHS * len(eval_dataloader)), desc="Evaluation Progress", leave=False)

# 创建历史记录字典
history = {
    'train_loss': [],
    'val_loss': [],
    'train_acc': [],
    'val_acc': []
}

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss_train = 0
    train_preds = []
    train_labels = []
    print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")

    for batch_num, batch in enumerate(train_dataloader):
        # (1)把数据移动到GPU
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        # (2)清除之前的梯度
        optimizer.zero_grad()
        # (3)前向传播
        outputs = model(**batch)
        loss = outputs.loss
        total_loss_train += loss.item()
        
        # (4)记录训练预测结果
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(batch["labels"].cpu().numpy())
        # (5)反向传播
        loss.backward()
        optimizer.step()
        # (6)更新学习率
        lr_scheduler.step()
        progress_bar_train.update(1)
        # (7)每50个batch记录Loss
        if (batch_num + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}, Batch {batch_num+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

    avg_train_loss = total_loss_train / len(train_dataloader)
    print(f"  Average Training Loss in Epoch {epoch + 1}: {avg_train_loss:.4f}")
    
    # 计算训练准确率
    train_accuracy = accuracy_score(train_labels, train_preds)
    avg_train_loss = total_loss_train / len(train_dataloader)
    print(f"  Average Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
    
    # 每训练一轮的同时验证一轮
    model.eval()
    total_loss_eval = 0
    val_preds = []
    val_labels = []
    
    with torch.no_grad(): # 测试的时候不需要计算梯度
        for batch in eval_dataloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss_eval += loss.item()
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(batch["labels"].cpu().numpy())
            progress_bar_eval.update(1)
    
    val_accuracy = accuracy_score(val_labels, val_preds)
    avg_val_loss = total_loss_eval / len(eval_dataloader)
    print(f"  Evaluation Loss: {avg_val_loss:.4f}, Evaluation Accuracy: {val_accuracy:.4f}")
    
    # 记录历史
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['train_acc'].append(train_accuracy)
    history['val_acc'].append(val_accuracy)
    
    progress_bar_eval.reset() # 为下一轮验证重置

progress_bar_train.close()
progress_bar_eval.close()
print("Training complete.")

# 5. 模型测试
print("\nFinal evaluation on the test dataset: ")
model.eval()
all_preds_final = []
all_labels_final = []
final_eval_progress = tqdm(eval_dataloader, desc="Final Evaluation")

with torch.no_grad():
    for batch in final_eval_progress:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        all_preds_final.extend(predictions.cpu().numpy())
        all_labels_final.extend(batch["labels"].cpu().numpy())

final_accuracy = accuracy_score(all_labels_final, all_preds_final)
print(f"Test Accuracy: {final_accuracy:.4f}")

# 6. 保存微调的模型以及分词配置
print(f"\nSaving model to {OUTPUT_DIR}")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Model and tokenizer saved.")

# 7. 绘制学习曲线
print("\nGenerating training visualization:")

# 创建图表
plt.figure(figsize=(14, 5))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Training Loss', marker='o', color='blue')
plt.plot(history['val_loss'], label='Validation Loss', marker='o', color='red')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(range(NUM_EPOCHS), [str(i+1) for i in range(NUM_EPOCHS)])
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Training Accuracy', marker='o', color='blue')
plt.plot(history['val_acc'], label='Validation Accuracy', marker='o', color='red')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(range(NUM_EPOCHS), [str(i+1) for i in range(NUM_EPOCHS)])
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# 调整布局并保存图片
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/training_history.png", dpi=300, bbox_inches='tight')
print(f"Training visualization saved to {OUTPUT_DIR}/training_history.png")