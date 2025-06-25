# Тренировка YOLO11

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

Создаем набор данных и обучаем YOLO 11 на основе этого пользовательского набора данных и обсуждаем результаты и показатели
## Features

- Создание набора данных из видео
- Создайте и обучите YOLO 11 для нескольких итераций
- Обсудите результаты и сделайте выводы

## Screenshots
# Видеофайлы 
![image](https://github.com/user-attachments/assets/a7bb43d7-8bfc-4660-871c-bf4efeb479c0)
# Набор данных 
![image](https://github.com/user-attachments/assets/2c03bcbe-958f-4d1d-a90d-e2b32cd69d5d)
# Построение и обучение модели yolo11 
![image](https://github.com/user-attachments/assets/e5ae52a9-6a74-49cc-bc87-114c2de597f9)
 
# Результаты 
![image](https://github.com/user-attachments/assets/f5eeb9b4-78a8-4f45-be44-8a41a422f1af)


## Installation

# 1-Mount your goole drive
```bash
from google.colab import drive
drive.mount('/content/drive')
```

# 2-Create structure of dataset
```bash
#creating folders
import os
dataset_dir = '/content/drive/My Drive/Dataset/'
raw_img_dir   = dataset_dir+'raw_img_1_MOV/'
raw_lbl_dir   = dataset_dir+'raw_lbl_1_MOV'

classes_file_path= dataset_dir+'classes.txt'
os.makedirs(raw_img_dir, exist_ok=True)
os.makedirs(raw_lbl_dir, exist_ok=True)
```

# 3- Create Classes file
```bash
Classes=list()
Classes.append("Dinner Table") #0
Classes.append("Tea Glass") #1
Classes.append("Spoon Box")#2
Classes.append("Person")#3
Classes.append("Tea Jar")#4
Classes.append("Human Hand") #5

index=0;
with open(classes_file_path, 'w') as writefile:
  while index<6:
    line=Classes[index]+"\n";
    writefile.write(line);
    index+=1;
```

# 4-Create Yaml dataset configuration file
```bash
# data.yaml - Dataset configuration

# Paths (relative to this file or absolute)
path: /content/drive/My Drive/Dataset/  # root dataset directory
train: images/train  # training images
val: images/val      # validation images
test: images/test    # testing images (optional)
nc: 6
# Class definitions
names: ['Dinner Table', 'Tea Glass', 'Spoon Box','Person','Tea Jar','Human Hand']



# info
author: Ahmad Ismail
date_created: 2025-06-24
version: 1.0
description: Custom object detection dataset 
```

# 5- Build dataset (extracting Frames and building annotation and bounding boxes)

# For fixed  background we manually determine classes and locations
Note we used YOLO TXT format for annotations and classes IDs
```bash
  import os
  import cv2

  video_path = '/content/drive/My Drive/Video/1.MOV'
  cap = cv2.VideoCapture(video_path)

  imgs_dir = raw_img_dir
  lbls_dir = raw_lbl_dir
  frame_count = 0

  # first video has fixed set of objects (table,tea glass, spoon box)
  # so for every frame add lables and boxes for these objects
  while True:
       ret, frame = cap.read()
       if not ret:
           break

       H, W = frame.shape[:2]

       print(height, width )

       # Save frame
       image_filename = os.path.join(imgs_dir, f'frame_{frame_count:05d}.jpg')  # image file
       lbl_filename   = os.path.join(lbls_dir, f'frame_{frame_count:05d}.txt')  # label file according YOLO TXT format
       cv2.imwrite(image_filename, frame)
       with open(lbl_filename, 'w') as writefile:
          ## class_id, center_x,center_y,with,hieght  normalized values
          writefile.write("0 0.58 0.5  0.85  0.7\n")
          writefile.write("1 0.32 0.19  0.08  0.12\n")
          writefile.write("1 0.32 0.76  0.08  0.12\n")
          writefile.write("2 0.165 0.3  0.07  0.28\n")

       frame_count += 1
       print(f"Processed {frame_count} frames")
  cap.release()

```
# For moving objects
```bash
# moving tea jar and hand
frame_count=102
while frame_count<=178 :
   lbl_filename   = os.path.join(lbls_dir, f'frame_{frame_count:05d}.txt')  # label file according YOLO TXT format
   with open(lbl_filename, 'w') as writefile:
          ## class_id, center_x,center_y,with,hieght  normalized values
          writefile.write("0 0.58 0.5  0.85  0.7\n")
          writefile.write("1 0.32 0.19  0.08  0.12\n")
          writefile.write("1 0.32 0.76  0.08  0.12\n")
          writefile.write("2 0.165 0.3  0.07  0.28\n")
          x=0.035+0.022*frame_count
          if x> 0.52:
            x=0.52
          y=0.16+0.04*frame_count
          if y > 0.4:
              y=0.4
          line="4 "+str(x)+" "+str(y)+" 0.067  0.16\n";
          writefile.write(line)
          if frame_count>112 :
            if frame_count<144:
              writefile.write("5 0.15 0.26 0.14  0.08\n")
            elif frame_count <156:
               writefile.write("5 0.15 0.2 0.14  0.08\n")

   frame_count+=1;
```
# 6- Building Train/val/Test dataset
```bash
import os
import random
import shutil
from sklearn.model_selection import train_test_split


images_dir = raw_img_dir
labels_dir = raw_lbl_dir
output_dir = dataset_dir

# Create output directories
for folder in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_dir, 'images', folder), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', folder), exist_ok=True)

# Get list of image files (assuming .jpg, modify if different)
image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

# Split into train+temp (80%), then split temp into val and test (50% each)
train_files, temp_files = train_test_split(image_files, test_size=0.2, random_state=42)
val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

# Copy Files into dataset
def copy_files(files, split_name):
   
    for file in files:
        # Copy image
        shutil.copy(
            os.path.join(images_dir, file),
            os.path.join(output_dir, 'images', split_name, file)
        )

        # Copy corresponding label file (assuming .txt format)
        label_file = os.path.splitext(file)[0] + '.txt'
        shutil.copy(
            os.path.join(labels_dir, label_file),
            os.path.join(output_dir, 'labels', split_name, label_file)
        )

# Copy files to their respective directories
copy_files(train_files, 'train')
copy_files(val_files, 'val')
copy_files(test_files, 'test')

# Print statistics
print(f"Total images: {len(image_files)}")
print(f"Train: {len(train_files)} ({len(train_files)/len(image_files):.1%})")
print(f"Validation: {len(val_files)} ({len(val_files)/len(image_files):.1%})")
print(f"Test: {len(test_files)} ({len(test_files)/len(image_files):.1%})")
```

# 7- Building/Training YOLO11
```bash
!pip install ultralytics
from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model
results = model.train(
    data= dataset_dir+'Dinner_dataset.yaml',
    epochs=60,
    imgsz=640,
    batch=16,
    device='cpu',
    val=True,       # Enable validation
    plots=True,    # Generate metric plots
    save_json=True  # Save metrics to JSON
)
```
# 8- Printing Results
```bash
import pandas as pd
from google.colab import files
from IPython.display import display


# Read and display CSV
results_df = pd.read_csv('/content/runs/detect/train19/results.csv')  # or next(iter(uploaded.keys()))
display(results_df.head())  # Shows first 5 rows with nice formatting

# For full table with scroll
display(results_df.style.set_properties(**{'text-align': 'center'}).set_table_styles(   [dict(selector='th', props=[('text-align', 'center')])]))
```
# 9- Printing Metrics
```bash
mport matplotlib.pyplot as plt
import matplotlib.image as mpimg
res_dir='/content/runs/detect/train19/'
# List of YOLO result images you might have
result_images = [
    'results.png',          # Training metrics plot
    'confusion_matrix.png', # Confusion matrix
    'F1_curve.png',        # F1-score curve
    'P_curve.png',         # Precision curve
    'R_curve.png'          # Recall curve
]

# Display each image that exists
for img_file in result_images:
    try:
        plt.figure(figsize=(10, 6))
        img = mpimg.imread(res_dir+img_file)
        plt.imshow(img)
        plt.title(img_file.split('.')[0].replace('_', ' ').title())
        plt.axis('off')
        plt.show()
    except FileNotFoundError:
        print(f"Note: {img_file} not found ")
```
