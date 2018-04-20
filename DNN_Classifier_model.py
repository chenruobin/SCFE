import tensorflow as tf
import robin_csv_reader as rd
import numpy as np
#定义特征目录
feature_columns = [tf.feature_column.numeric_column("x", shape=[14])]
#定义分类器
classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 20, 10],
                                            n_classes=2,
                                            model_dir="./modelSaver")
#读取数据
x, y = rd.read_csv_data('SCFE.csv', 15, 14)
#设置输入格式
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(x)},
    y=np.array(y,dtype=int),
    num_epochs=None,
    shuffle=True)
#训练
classifier.train(input_fn=train_input_fn, steps=2000)
