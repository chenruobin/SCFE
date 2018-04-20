import tensorflow as tf
import numpy as np
import robin_csv_reader as rd
print("test start")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    feature_columns = [tf.feature_column.numeric_column("x", shape=[14])]
    # 定义分类器
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 20, 10],
                                            n_classes=2,
                                            model_dir="./modelSaver")
    # new_samples = np.array(
    #     [[11, 10.6, 2.51, 1.11, 21.15, 9.74, 35.4, 463, 189, 17.2, 28.3, 21.4, 10.7, 22.4],
    #      [13, 14.9, 1.6, 0.636, 42.91, 27.12, 73.7, 460, 174, 22.6, 31.3, 23.6, 11.6, 10.9]], dtype=np.float32)
    # predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"x": new_samples},
    #     num_epochs=1,
    #     shuffle=False)
    #
    # predictions = list(classifier.predict(input_fn=predict_input_fn))
    # predicted_classes = [p["classes"] for p in predictions]
    #
    # print(
    #     "New Samples, Class Predictions:    {}\n"
    #         .format(predicted_classes))
    # 读取数据
    x, y = rd.read_csv_data('SCFE_test.csv', 15, 14)

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":  np.array(x)},
        num_epochs=1,
        shuffle=False)

    predictions = list(classifier.predict(input_fn=predict_input_fn))
    predicted_classes = [p["classes"] for p in predictions]
    print("评估结果为：", np.array(predicted_classes,dtype=int).reshape(12))
    print("正确结果为：", np.array(y, dtype=int))
    # 设置输入格式
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(x)},
        y=np.array(y, dtype=int),
        num_epochs=1,
        shuffle=False)


    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))