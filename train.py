import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import regularizers

datagen = ImageDataGenerator(
    rotation_range=5,  # 减小旋转幅度
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,  # 保留水平翻转
    brightness_range=[0.9, 1.1]  # 减小亮度变化
)

def process_images(main_folder):
    images=[]
    labels=[]
    filesname = sorted([d for d in os.listdir(main_folder)
                        if os.path.isdir(os.path.join(main_folder, d))])
    files_label = {"male": 0, "female": 1}  # 强制指定标签

    for name in filesname :
        folders_name=os.path.join(main_folder,name)
        if not os.path.isdir(folders_name) :
            continue

        for image_name in os.listdir(folders_name) :
            image_path=os.path.join(folders_name,image_name)
            image=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
            print(f"正在处理第{name}个文件夹的第{image_name}张图片")

            if image is not None :  # 读取成功
                image_resize = cv2.resize(image, (128, 128))    # 缩放
                images.append(image_resize) # 添加到列表
                labels.append(files_label[name]) # 添加标签

                image_resize=np.expand_dims(image_resize,axis=-1)
                image_resize = np.expand_dims(image_resize, axis=0)

                for _ in range(5):  # 减少数据增强次数  # 5次
                    image_aug = datagen.flow(image_resize, batch_size=1)
                    image_augmented = next(image_aug)[0].astype(np.uint8).squeeze()
                    images.append(image_augmented)
                    labels.append(files_label[name])


    return np.array(images),np.array(labels)

images,labels=process_images(r"database")

images=images/255.0
image=images.reshape(images.shape[0],128,128,1)

x_train,x_test,y_train,y_test=train_test_split(image,labels,test_size=0.1,random_state=42) # 划分数据集
train_generator=datagen.flow(x_train,y_train,batch_size=32) # 生成器

def cnn_model() :
    model=models.Sequential([
        layers.Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(128,128,1),
                      kernel_regularizer=regularizers.l2(0.001)),
        BatchNormalization(),
        layers.MaxPooling2D((2,2),strides=2),

        layers.Conv2D(64,(3,3),activation='relu',padding='same',
                      kernel_regularizer=regularizers.l2(0.001)),
        BatchNormalization(),
        layers.MaxPooling2D((2,2),strides=2),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3),activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(0.001)),
        BatchNormalization(),
        layers.MaxPooling2D((2, 2), strides=2),

        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(256,activation='relu'),
        layers.Dense(1, activation='sigmoid')  # 二分类方案
    ])
    return model

model=cnn_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',  # 二分类损失函数
              metrics=['accuracy'])

early_stopping = EarlyStopping(
    monitor='val_loss', # 监控验证集损失
    patience=5, # 5次没有改善就停止
    restore_best_weights=True # 恢复最佳权重
)   # 提前停止

history=model.fit(
    x_train,y_train,
    batch_size=128,
    epochs=10,
    validation_data=(x_test,y_test),
    callbacks=[early_stopping]
)

test_loss,test_acc=model.evaluate(x_test,y_test)
print(f"accurancy is {test_acc},loss is {test_loss}")

model.save("gender_1.h5")