import pandas as pd
import os
import re
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import RobertaTokenizer
from collections import Counter
import unicodedata
import matplotlib.pyplot as plt
import seaborn as sns
import torch
print(torch.cuda.is_available())  # باید True برگردونه

# ==========================================
# 📂 ۱. تنظیم مسیرها و بارگذاری داده
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, 'data')

test = pd.read_parquet(os.path.join(data_path, 'test-00000-of-00001.parquet'))
train = pd.read_parquet(os.path.join(data_path, 'train-00000-of-00001 (1).parquet'))
val = pd.read_parquet(os.path.join(data_path, 'validation-00000-of-00001.parquet'))

print(f'num train data is {len(train)}')
cols = train.columns.to_list()
print(cols, len(cols))
print(train.info())

# ==========================================
# 🧹 ۲. توابع تمیزسازی
# ==========================================
def clean_Data(data):
    if not isinstance(data, str):
        return ''
    
    data = re.sub(r'http\S+|https\S+', '', data)
    data = unicodedata.normalize('NFKD', data).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    data = re.sub(r'[^a-zA-Z0-9\s\.\,\!\%\?\$\'\"\)\(]', ' ', data)
    data = re.sub(r'\s+', ' ', data).strip()
    data = data.lower()
    
    return data

# ==========================================
# 🧹 ۳. اعمال تمیزسازی روی داده
# ==========================================
train['text'] = train['text'].apply(clean_Data)
test['text'] = test['text'].apply(clean_Data)
val['text'] = val['text'].apply(clean_Data)

# ==========================================
# 📊 ۴. تحلیل توزیع طول توکن‌ها (قبل از توکنایز)
# ==========================================
print("\n📊 در حال تحلیل توزیع طول توکن‌ها...")

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def count_tokens(text):
    return len(tokenizer.encode(text, truncation=False))

sample_size = 43410
train_sample = train.head(sample_size)
train_sample['token_length'] = train_sample['text'].apply(count_tokens)

print("\n📊 آمار توصیفی طول توکن‌ها:")
print(train_sample['token_length'].describe())

percentiles = [50, 75, 90, 95, 99, 99.9, 100]
print("\n📈 صدک‌های مهم:")
for p in percentiles:
    value = np.percentile(train_sample['token_length'], p)
    print(f"   صدک {p}: {value:.0f} توکن")

over_128 = (train_sample['token_length'] > 128).sum()
percent_over_128 = (over_128 / len(train_sample)) * 100
print(f"\n📌 درصد نمونه‌هایی که بیش از ۱۲۸ توکن دارند: {percent_over_128:.2f}%")

over_256 = (train_sample['token_length'] > 256).sum()
percent_over_256 = (over_256 / len(train_sample)) * 100
print(f"📌 درصد نمونه‌هایی که بیش از ۲۵۶ توکن دارند: {percent_over_256:.2f}%")

plt.figure(figsize=(12, 6))
sns.histplot(train_sample['token_length'], bins=50, kde=True)
plt.axvline(x=128, color='red', linestyle='--', label='max_length=128')
plt.axvline(x=256, color='orange', linestyle='--', label='max_length=256')
plt.xlabel('تعداد توکن‌ها')
plt.ylabel('تعداد نمونه‌ها')
plt.title('توزیع طول توکن‌ها در دیتاست GoEmotions')
plt.legend()
plt.grid(True)
plt.savefig('./token_length_distribution.png', dpi=300)
plt.show()

print("\n💡 نتیجه‌گیری:")
if percent_over_128 < 5:
    print("✅ max_length=128 انتخاب بسیار مناسبی است.")
elif percent_over_128 < 10:
    print("⚠️ max_length=128 نسبتاً مناسب است، اما ممکن است ۲۵۶ بهتر باشد.")
else:
    print("❌ پیشنهاد می‌شود max_length=256 تنظیم شود.")

# ==========================================
# 🔄 ۵. تبدیل به DatasetDict و توکنایز
# ==========================================
train = Dataset.from_pandas(train[train['text'].str.len()>0])
test = Dataset.from_pandas(test[test['text'].str.len()>0])
val = Dataset.from_pandas(val[val['text'].str.len()>0])

dataset = DatasetDict({
    'train': train,
    'test': test,
    'validation': val
})

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset = tokenized_dataset.rename_column("labels", "label")
tokenized_dataset.set_format("torch")

print('saving tokenized data...')
tokenized_dataset.save_to_disk('./tokenized')
print('tokenized data saved successfully.')

print("✅ آماده‌سازی داده‌ها کامل شد!")
print(f"تعداد نمونه‌های آموزش: {len(tokenized_dataset['train'])}")
print(f"تعداد نمونه‌های اعتبارسنجی: {len(tokenized_dataset['validation'])}")
print(f"تعداد نمونه‌های تست: {len(tokenized_dataset['test'])}")