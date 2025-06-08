


# i have the data into folder such as :
# {'0': 'specification', '1': 'scientific_report', '2': 'scientific_publication', '3': 'resume', '4': 'questionnaire',
#             '5': 'presentation',  '6': 'news_article', '7': 'memo', 
#             '8': 'letter', '9': 'invoice', '10': 'handwritten','11': 'form', 
#             '12': 'file_folder', '13': 'email', '14': 'budget', '15': 'advertisement'}







import os
import shutil
import random
from pathlib import Path

# Original dataset directory (with folders '0', '1', ..., '15')
source_dir = "test"
train_dir = "dataset_split/train_df"
test_dir = "dataset_split/test_df"

# Split ratio
test_ratio = 0.2  # 20% test, 80% train

# Ensure output directories exist
for split_dir in [train_dir, test_dir]:
    os.makedirs(split_dir, exist_ok=True)

# Set seed for reproducibility
random.seed(42)

# Loop through each class folder
for class_id in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_id)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    split_point = int(len(images) * (1 - test_ratio))
    train_images = images[:split_point]
    test_images = images[split_point:]

    # Create class subfolder in train and test directories
    train_class_dir = os.path.join(train_dir, class_id)
    test_class_dir = os.path.join(test_dir, class_id)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    # Copy train images
    for img in train_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(train_class_dir, img)
        shutil.copy2(src, dst)

    # Copy test images
    for img in test_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(test_class_dir, img)
        shutil.copy2(src, dst)

print("✅ Dataset split complete.")


