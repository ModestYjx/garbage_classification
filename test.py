import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers, models,optimizers,losses
from    tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.001,
    patience=5
)

# 数据预处理
def processing_data(data_path):
    """
    数据处理
    :param data_path: 数据集路径
    :return: train, test:处理后的训练集数据、测试集数据
    """
    # -------------------------- 实现数据处理部分代码 ----------------------------
    # 图片的长宽
    height, width = 384, 512
    # 训练数据：测试数据 = 8：2
    validation_split = 0.2
    # 批量处理数据大小
    batch_size = 16

    img_list = glob.glob(os.path.join(data_path, '*/*.jpg'))
    print("数据集数量：", len(img_list))
    train_datagen = ImageDataGenerator(
        # 对图片的每个像素值均乘上这个放缩因子，把像素值放缩到0和1之间有利于模型的收敛
        rescale=1. / 255,
        # 浮点数，剪切强度（逆时针方向的剪切变换角度）
        shear_range=0.1,
        # 随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]
        zoom_range=0.1,
        # 浮点数，图片宽度的某个比例，数据提升时图片水平偏移的幅度
        width_shift_range=0.1,
        # 浮点数，图片高度的某个比例，数据提升时图片竖直偏移的幅度
        height_shift_range=0.1,
        # 布尔值，进行随机水平翻转
        horizontal_flip=True,
        # 布尔值，进行随机竖直翻转
        vertical_flip=True,
        # 在 0 和 1 之间浮动。用作验证集的训练数据的比例
        validation_split=validation_split
    )
    test_datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=validation_split
    )

    train_generator = train_datagen.flow_from_directory(
        # 提供的路径下面需要有子目录
        data_path,
        # 整数元组 (height, width)，默认：(256, 256)。 所有的图像将被调整到的尺寸。
        target_size=(height, width),
        # 一批数据的大小
        batch_size=batch_size,
        # "categorical", "binary", "sparse", "input" 或 None 之一。
        # 默认："categorical",返回one-hot 编码标签。
        class_mode='categorical',
        # 数据子集 ("training" 或 "validation")
        subset='training',
        seed=0
    )

    validation_generator = test_datagen.flow_from_directory(
        data_path,
        target_size=(height, width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        seed=0)

    # ------------------------------------------------------------------------
    train_data, test_data = train_generator, validation_generator
    return train_data, test_data

def model(train_data, test_data, save_model_path):
    """
    创建、训练和保存深度学习模型
    :param train_data: 训练集数据
    :param test_data: 测试集数据
    :param save_model_path: 保存模型的路径和名称
    :return:
    """
    # --------------------- 实现模型创建、训练和保存等部分的代码 ---------------------

    model = Sequential([
        layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(384, 512, 3)),
        layers.MaxPooling2D(pool_size=2),

        layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=2),

        layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=2),

        layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=2),

        layers.Flatten(),

        layers.Dense(64, activation='relu'),

        layers.Dense(6, activation='softmax')
    ])

    model.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.Adam(1e-4), metrics=['acc'])

    model.summary()
    #     使用VGG19迁移学习
    #     net = keras.applications.VGG19(
    #         weihts='vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
    #         include_top = False,
    #         pooling='max'
    #     )
    #     net.trainable = False
    #     new_net = keras.Sequential([
    #         net,
    #         layers.Dense(6)
    #     ])
    #     new_net.build(input_shape=())

    # 保存模型（请写好保存模型的路径及名称）
    # -------------------------------------------------------------------------

    # model.fit_generator(train_data,epochs=20,validation_data=test_data)

    model.fit(train_data, validation_data=test_data, validation_freq=1, epochs=100,
               callbacks=[early_stopping])

    # 保存模型
    model.save(save_model_path)
    return model


def evaluate_mode(test_data, save_model_path):
    """
    加载模型和评估模型
    可以实现，比如: 模型训练过程中的学习曲线，测试集数据的loss值、准确率及混淆矩阵等评价指标！
    主要步骤:
        1.加载模型(请填写你训练好的最佳模型),
        2.对自己训练的模型进行评估

    :param test_data: 测试集数据
    :param save_model_path: 加载模型的路径和名称,请填写你认为最好的模型
    :return:
    """
    # ----------------------- 实现模型加载和评估等部分的代码 -----------------------

    model = models.load_model(save_model_path)

    # ---------------------------------------------------------------------------


def main():
    """
    深度学习模型训练流程,包含数据处理、创建模型、训练模型、模型保存、评价模型等。
    如果对训练出来的模型不满意,你可以通过调整模型的参数等方法重新训练模型,直至训练出你满意的模型。
    如果你对自己训练出来的模型非常满意,则可以提交了!
    :return:
    """
    data_path = "./datasets/la1ji1fe1nle4ishu4ju4ji22-momodel/dataset-resized/"  # 数据集路径
    save_model_path = None  # 保存模型路径和名称

    # 获取数据
    train_data, test_data = processing_data(data_path)

    # 创建、训练和保存模型
    model(train_data, test_data, save_model_path)

    # 评估模型
    evaluate_mode(test_data, save_model_path)


if __name__ == '__main__':
    main()