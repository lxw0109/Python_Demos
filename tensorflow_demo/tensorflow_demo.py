#!/usr/bin/env python3
# coding: utf-8
# File: tensorflow_demo.py
# Author: lxw
# Date: 6/11/18 10:47 AM

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class TF_Examples:
    """
    Reference: [TensorFlow入门一](https://zhuanlan.zhihu.com/p/30487008)
    """
    def __init__(self):
        self.learning_rate = 1e-2

    def basics(self):
        """
        Reference: [五分钟带你入门TensorFlow](https://www.jianshu.com/p/2ea7a0632239)
        :return: 
        """
        # 1. 简单的矩阵乘法
        v1 = tf.constant([[2, 3]])  # 1行2列
        v2 = tf.constant([[2], [3]])  # 2行1列
        product = tf.matmul(v1, v2)  # 此时不会立即执行，需要在会话中执行才行
        print(product)

        sess = tf.Session()
        result = sess.run(product)
        print(result)
        sess.close()

        # 2. 创建一个变量，并用for循环对变量进行赋值操作
        num = tf.Variable(0, name="count")
        new_value = tf.add(num, 10)
        op = tf.assign(num, new_value)  # tf.assign(dest, src): 赋值操作

        with tf.Session() as sess:
            # 初始化变量
            sess.run(tf.global_variables_initializer())  # Essential, otherwise "tensorflow.python.framework.errors_impl.FailedPreconditionError: Attempting to use uninitialized value count"
            print(sess.run(num))  # 0
            for i in range(5):
                sess.run(op)
                print(sess.run(num))  # 10, 20, ...
        """
        # num = tf.constant(10)
        num = tf.Variable(10)
        num = tf.add(num, 1)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(5):
                print(sess.run(num))  # 11, 11, 11, 11, 11
        """

        # 3. 通过feed设置placeholder的值
        input1 = tf.placeholder(tf.float32)
        input2 = tf.placeholder(tf.float32)

        new_value = tf.multiply(input1, input2)

        with tf.Session() as sess:
            result = sess.run(new_value, feed_dict={input1: 20.0, input2: 11.0})
            print(f"tf.multiply(input1, input2): {result}")


        # 4. 返回多个运行结果
        """
        Reference: [cs20si: tensorflow for research 学习笔记1](https://zhuanlan.zhihu.com/p/28488510)
        """
        x = 2
        y = 3
        add_op = tf.add(x, y)
        mul_op = tf.multiply(x, y)
        mul_add_op = tf.multiply(x, add_op)
        pow_op = tf.pow(add_op, mul_op)
        with tf.Session() as sess:
            mul_add_result, pow_result = sess.run([mul_add_op, pow_op])  # 调用sess.run时, 使用[]来得到多个结果
            print(f"mul_add_result: {mul_add_result}, pow_result: {pow_result}")

            # TypeError: Fetch argument 5 has invalid type <class "int">, must be a string or Tensor.
            # (Can not convert a int into a Tensor or Operation.)
            # print(sess.run(x + y))  # NO
            print(sess.run(tf.constant(x) + tf.constant(y)))  # OK

    def linear_regression_demo(self):
        epochs = 2000
        display_step = 50

        # Training Data
        X_train = np.array(
            [3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59,
             2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])  # <ndarray>. shape: (17,)
        y_train = np.array(
            [1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53,
             1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])  # <ndarray>. shape: (17,)
        n_samples = X_train.shape[0]  # 17

        # tf Graph Input
        X = tf.placeholder("float")
        Y = tf.placeholder("float")

        # Set model weights
        W = tf.Variable(np.random.randn(), name="weight")
        b = tf.Variable(np.random.randn(), name="bias")

        # Construct a linear model
        pred = tf.add(tf.multiply(X, W), b)  # pred: <Tensor>

        # MSE: Mean Squared Error
        cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * n_samples)  # cost: <Tensor>
        # Gradient Descent
        # NOTE: minimize() knows to modify W and b because Variable objects are trainable=True by default.
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)  # NOTE: optimizer: <Operation>

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # Start training
        with tf.Session() as sess:
            # Run the initializer
            sess.run(init)

            # 1. Fit all training data
            for epoch in range(epochs):
                for (x, y) in zip(X_train, y_train):
                    sess.run(optimizer, feed_dict={X: x, Y: y})

                # Display logs per epoch step
                if (epoch + 1) % display_step == 0:
                    c = sess.run(cost, feed_dict={X: X_train, Y: y_train})
                    print(f"Epoch:{epoch + 1:4d}, cost={c:.9f}, W={sess.run(W)}, b={sess.run(b)}")

            print("Optimization Finished!")
            training_cost = sess.run(cost, feed_dict={X: X_train, Y: y_train})
            print(f"Training cost={training_cost}, W={sess.run(W)}, b={sess.run(b)}\n")

            # Graphic display
            plt.plot(X_train, y_train, "ro", label="Original data")
            plt.plot(X_train, sess.run(W) * X_train + sess.run(b), "yo", label="Fitted points")
            plt.plot(X_train, sess.run(W) * X_train + sess.run(b), label="Fitted line")
            # sess.run(W) * X_train + sess.run(b): <ndarray>. shape (17,)
            plt.legend()
            plt.show()

            # 2. Testing example, as requested (Issue #2)
            X_test = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
            y_test = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

            print("Testing... (Mean square loss Comparison)")
            # sess.run(tf.pow(pred - Y, 2) / (2 * X_test.shape[0]), feed_dict={X: X_test, Y: y_test})  # <ndarray>
            testing_cost = sess.run(tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * X_test.shape[0]),
                                    feed_dict={X: X_test, Y: y_test})  # same function as cost above
            print(f"Testing cost={testing_cost}")
            print(f"Absolute mean square loss difference: {abs(training_cost - testing_cost)}")

            plt.plot(X_test, y_test, "bo", label="Testing data")
            plt.plot(X_train, sess.run(W) * X_train + sess.run(b), "ro",  label="Fitted point")
            plt.plot(X_train, sess.run(W) * X_train + sess.run(b), label="Fitted line")
            plt.legend()
            plt.show()

    def logistic_regression_demo(self):
        import tensorflow as tf

        # Import MNIST data
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

        # Parameters
        training_epochs = 25
        batch_size = 100
        display_step = 1

        # tf Graph Input
        x = tf.placeholder(tf.float32, [None, 784])  # mnist data image of shape 28*28=784. x: <Tensor>, shape: (?, 784)
        y = tf.placeholder(tf.float32, [None, 10])  # 0-9 digits recognition => 10 classes. y: <Tensor>, shape: (?, 10)

        # Set model weights
        W = tf.Variable(tf.zeros([784, 10]))  # shape: (784, 10)
        b = tf.Variable(tf.zeros([10]))  # shape: (10,)  # NOTE: 在tensorflow里, [1, 2, 3]的shape为(3,)，要理解为3列而不是3行

        # Construct model
        pred = tf.nn.softmax(tf.matmul(x, W) + b)  # pred: <Tensor>, shape: (?, 10)  # TODO: ? * 10 + 1 * 10.

        # Minimize error using cross entropy
        cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), axis=1))  # NOTE: y是矩阵(batch_size, 10),能直接用*运算? "*"相当于内积运算, 对应相乘 [[1, 2, 3], [2, 2, 2]] * [[4, 5, 6], [1, 2, 1]] => [[4, 10, 18], [2, 4, 2]]
        # sess.run(-tf.reduce_sum(y * tf.log(pred), axis=1), feed_dict={x: batch_xs, y: batch_ys}): shape: (batch_size,). axis=1, 理解为跨列(across axis 1)
        # Gradient Descent
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # Start training
        with tf.Session() as sess:
            # Run the initializer
            sess.run(init)

            # Training cycle
            for epoch in range(training_epochs):
                avg_cost = 0.
                total_batch = int(mnist.train.num_examples / batch_size)

                # Loop over all batches
                for i in range(total_batch):
                    # batch_xs: <ndarray of ndarray>, shape: (batch_size, 784) # batch_ys: <ndarray of ndarray>, shape: (batch_size, 10).
                    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})  # _: None
                    # Compute average loss
                    avg_cost += c / total_batch

                # Display logs per epoch step
                if (epoch + 1) % display_step == 0:
                    print(f"Epoch:{epoch + 1:4d}, cost={avg_cost:.9f}")

            print("Optimization Finished!")

            # Test model
            correct_prediction = tf.equal(tf.argmax(pred, axis=1), tf.argmax(y, axis=1))  # shape: (?,). type: <Tensor>. dtype: bool
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))  # 0.9138


if __name__ == "__main__":
    tf_eg = TF_Examples()

    # tf_eg.basics()

    tf_eg.linear_regression_demo()

    # tf_eg.logistic_regression_demo()
