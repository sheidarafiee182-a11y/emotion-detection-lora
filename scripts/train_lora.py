#1. import necessary modules 
import torch
import numpy as np
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from datasets import load_from_disk
import warnings
warnings.filterwarnings('ignore')



tokenized_data = load_from_disk('./tokenized')
print("🔄 data loded sucssesfullly.")
print(f"   train samples : {len(tokenized_data['train'])}")
print(f"   val samples: {len(tokenized_data['validation'])}")
print(f"   test samples: {len(tokenized_data['test'])}")


print("\n🔄 load base model(roberta-base)...")
model = RobertaForSequenceClassification.from_pretrained(
    'roberta-base',
    num_labels=28,
    ignore_mismatched_sizes=True
)
print(' model loaded')

# ==========================================
#  2. set lora settinggs
# ==========================================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=['query', 'value'],
    lora_dropout=0.1,
    bias='none',
    task_type=TaskType.SEQ_CLS
)

model = get_peft_model(model, lora_config)
print("lora settings applied")
model.print_trainable_parameters()

# ==========================================
#  3. compute clasess weights to use becuase the data is imbalanced
# ==========================================

all_labels = []
for item in tokenized_data['train']:
    label = item['label']
    if torch.is_tensor(label):
        label = label.tolist()
    all_labels.extend(label)

unique_labels = np.unique(all_labels)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=unique_labels,
    y=all_labels
)
class_weight_dict = {int(label): weight for label, weight in zip(unique_labels, class_weights)}

print(' class weights computed')

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # استفاده از آستانه ۰.۵ برای دیتاست چندبرچسبی
    predictions = (predictions > 0.5).astype(int)
    
    f1_micro = f1_score(labels, predictions, average='micro')
    f1_macro = f1_score(labels, predictions, average='macro')
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'accuracy': accuracy
    }

# ==========================================
#  ۶.custom Data Collator for multi label dataset 
# ==========================================
class MultiLabelDataCollator:
    def __init__(self, tokenizer, num_labels=28):
        self.tokenizer = tokenizer
        self.num_labels = num_labels

    def __call__(self, features):
        labels = []
        for feature in features:
            label = feature.pop("label")
            if torch.is_tensor(label):
                label = label.tolist()
            labels.append(label)
        
        batch_size = len(labels)
        padded_labels = torch.zeros((batch_size, self.num_labels), dtype=torch.float32)
        
        for i, label_list in enumerate(labels):
            for idx in label_list:
                if idx < self.num_labels:
                    padded_labels[i, idx] = 1.0
    
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = padded_labels
        
        return batch



training_args = TrainingArguments(
    output_dir="./emotion_model_lora",
    logging_dir="./logs",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=500,
    gradient_accumulation_steps=2,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    save_total_limit=2,
    fp16=False,  
    report_to="none",
    logging_steps=25,
)

print("hyperparameters were tuned ")


print("start training...")

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data['train'],
    eval_dataset=tokenized_data['validation'],
    compute_metrics=compute_metrics,
    data_collator=MultiLabelDataCollator(tokenizer, num_labels=28),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("✅ Trainer prepared")


trainer.train()
print("✅ Training completed successfully!")

# ==========================================
# save finall model
# ==========================================
print("saving final model...")

model.save_pretrained("./emotion_model_lora_final")
tokenizer.save_pretrained("./emotion_model_lora_final")



# ==========================================
# evaluate on test set
# ==========================================


test_results = trainer.evaluate(tokenized_data['test'])

print("final results on test set:")
print(f"  F1-Macro:  {test_results['eval_f1_macro']:.4f}")
print(f"  F1-Micro:  {test_results['eval_f1_micro']:.4f}")
print(f"  Accuracy:  {test_results['eval_accuracy']:.4f}")
print(f"  Loss:      {test_results['eval_loss']:.4f}")

