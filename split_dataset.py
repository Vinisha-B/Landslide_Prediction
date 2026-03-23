import os
import random
import shutil

source = "archive/"           # your dataset folder
train_dir = "dataset/train/"
test_dir = "dataset/test/"

categories = ["landslide", "no_landslide"]  # subfolders inside archive

for category in categories:
    path = os.path.join(source, category)
    images = os.listdir(path)
    random.shuffle(images)

    split = int(0.8 * len(images))  # 80% train, 20% test

    train_images = images[:split]
    test_images = images[split:]

    # Create train/test folders
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)

    # Copy images to train
    for img in train_images:
        shutil.copy(os.path.join(path, img),
                    os.path.join(train_dir, category, img))

    # Copy images to test
    for img in test_images:
        shutil.copy(os.path.join(path, img),
                    os.path.join(test_dir, category, img))

print("✅ Dataset split completed!")