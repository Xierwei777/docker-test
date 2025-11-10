import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
random_seed = 66
dataset = 'data\pose_landmarks.csv'
model_save_path = 'pose_model.keras'
NUM_CLASSES = 5
TIME_STEPS = 16
DIMENSION = 2
# 使用pandas读取CSV文件，处理编码问题
df = pd.read_csv(dataset, encoding='utf-8')
X_dataset = df.iloc[:, 1:(33 * 2) + 1].values.astype('float32')
y_dataset = df.iloc[:, 0].values.astype('int32')
X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=random_seed)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input((33*2,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.summary()

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    model_save_path, verbose=1, save_weights_only=False)#模型检查点回调
es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)#提前停止训练
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(
    X_train,
    y_train,
    epochs=1000,
    batch_size=128,
    validation_data=(X_test, y_test),
    callbacks=[cp_callback, es_callback]
)
val_loss, val_acc = model.evaluate(X_test, y_test, batch_size=128)#分析报告
model = tf.keras.models.load_model(model_save_path)#保存
predict_result = model.predict(np.array([X_test[0]]))#进行预测，输出[[a,b,c,d]]分别为各种分类的概率
print(np.squeeze(predict_result))#降维
print(np.argmax(np.squeeze(predict_result)))#输出最大概率的标签


def print_confusion_matrix(y_true, y_pred, report=True):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)

    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df_cmx, annot=True, fmt='g', square=False)
    ax.set_ylim(len(set(y_true)), 0)
    plt.show()

    if report:
        print('Classification Report')
        print(classification_report(y_true, y_pred))


Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)

print_confusion_matrix(y_test, y_pred)
model.save(model_save_path, include_optimizer=False)
tflite_save_path = 'pose_model.tflite'

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]#开启默认优化，减少模型大小 & 提高推理速度。
tflite_quantized_model = converter.convert()#转换模型为 TFLite（量化）

open(tflite_save_path, 'wb').write(tflite_quantized_model)#模型保存
#再次调用.tflite模型
interpreter = tf.lite.Interpreter(model_path=tflite_save_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]['index'], np.array([X_test[0]]))
interpreter.invoke()
tflite_results = interpreter.get_tensor(output_details[0]['index'])
print(np.squeeze(tflite_results))
print(np.argmax(np.squeeze(tflite_results)))