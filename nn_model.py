import sys, os
import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(parent_dir, "data")
train_dir = os.path.join(parent_dir, "ckpt")
model_dir = os.path.join(parent_dir, "model/nn")

flags = tf.flags
flags.DEFINE_string("data_path", data_path, "data_path")
flags.DEFINE_string("train_dir", train_dir, "train_dir")
flags.DEFINE_string("model_dir", model_dir, "model_dir")

FLAGS = flags.FLAGS

class TensorNameConfig(object):
    input_tensor = "inputs"
    target_tensor = "target"
    output_tensor = "output_node"

def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

class NetworkModel(object):
    def inference(self, input_tensor):
        # Weight initializations
        self.w_1 = init_weights((self.layers[0], self.layers[1]))
        self.w_2 = init_weights((self.layers[1], self.layers[2]))
        
        # forward propagation
        h = tf.nn.sigmoid(tf.matmul(input_tensor, self.w_1))                                 # hidden layer
        self.logits = tf.nn.softmax(tf.matmul(h, self.w_2), name = self.conf.output_tensor)    # softmax logits of this example
        
        # Get prediction label
        self.y_predict = tf.argmax(self.logits, axis=1)

        self.saver = tf.train.Saver(tf.global_variables())

        return self.y_predict
        

    """ Simple Feed Forward Network with node defined in layers
    """
    def __init__(self, layers):
        self.layers = layers
        if (len(layers) != 3):
            print ("Input layer structure doesn't equal 3")
        
        self.x_size = layers[0]
        self.y_size = layers[len(layers) - 1]  # iris outcome 3 classes
        
        # New tensorname config
        self.conf = TensorNameConfig()

    def train(self):
        # placeholders
        self.X = tf.placeholder("float", shape=[None, self.x_size], name = self.conf.input_tensor)
        self.Y = tf.placeholder("float", shape=[None, self.y_size], name = self.conf.target_tensor)
        self.inference(self.X)
        # Backward propagation
        self.cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y, logits=self.logits))
        self.train_op = tf.train.GradientDescentOptimizer(0.01).minimize(self.cost)


def train():
    train_X, test_X, train_y, test_y = get_iris_data()

    # Layer's sizes
    x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
    h_size = 64                # Number of hidden nodes
    y_size = train_y.shape[1]   # Number of outcomes (3 iris flowers)
    layers = [x_size, h_size, y_size]
    print ("NN network layer structure: " + str(layers))

    tf.global_variables_initializer()

    with tf.Session() as session:
        # New Model
        model = NetworkModel(layers) # default network model
        model.train()
        #initialization
        session.run(tf.global_variables_initializer())

        for epoch in range(100):
            # See all the examples
            for x, y in zip(train_X, train_y):
                x = np.reshape(x, (1, x.shape[0]))  # shape[1, size]
                y = np.reshape(y, (1, y.shape[0]))
                fetch_list = [model.train_op, model.cost]  # [cost, train_op]
                session.run(fetch_list, feed_dict={model.X: x, model.Y: y})
            # evaluation
            if (epoch % 5 == 0):
                train_accuracy = np.mean(np.argmax(train_y, axis=1) == session.run(model.y_predict, feed_dict={model.X: train_X, model.Y: train_y}))
                test_accuracy = np.mean(np.argmax(test_y, axis=1) == session.run(model.y_predict, feed_dict={model.X: test_X, model.Y: test_y}))
                print ("Epoch %d, training acc %f and test accuracy %f" % (epoch, train_accuracy, test_accuracy))

            # Saving checkpoint file
            if (epoch % 20 == 0):
                checkpoint_path = os.path.join(FLAGS.train_dir, "nn_model.ckpt")
                model.saver.save(session, checkpoint_path)
                print("Model Saved... at epoch" + str(epoch))

        # write graph
        #tf.train.write_graph(session.graph_def, FLAGS.model_dir, "nn_model.pbtxt", as_text=True)
        #tf.train.write_graph(session.graph_def, FLAGS.model_dir, "nn_model.pb", as_text=False)
        tf.train.Saver().save(session, FLAGS.model_dir + "/model.ckpt")
    session.close()

def get_iris_data():
    """ Read the iris data set and split them into training and test sets """
    iris   = datasets.load_iris()
    data   = iris["data"]
    target = iris["target"]

    # Prepend the column of 1s for bias
    N, M  = data.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = data

    # Convert into one-hot vectors
    num_labels = len(np.unique(target))
    all_Y = np.eye(num_labels)[target]
    return train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)

def predict():
    train_X, test_X, train_y, test_y = get_iris_data()
    # Layer's sizes
    x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
    h_size = 64                # Number of hidden nodes
    y_size = train_y.shape[1]   # Number of outcomes (3 iris flowers)
    layers = [x_size, h_size, y_size]
    print ("NN network layer structure: " + str(layers))

    tf.global_variables_initializer()

    model = NetworkModel(layers) # default network model
    x = tf.placeholder("float", shape=[None, x_size], name = model.conf.input_tensor)

    with tf.Session() as session:
        # New Model
        model.inference(x)
        #initialization
        model.saver.restore(session, FLAGS.model_dir + "/model.ckpt")

        '''
        print("model.w_1")
        print(session.run(model.w_1))
        print("model.w_2")
        print(session.run(model.w_2))
        '''

        feed_dict = {x:[[1., 5.8, 4., 1.2, 0.2]]}
        result = session.run(model.logits, feed_dict=feed_dict);
        print("logits: " + str(result))
        result = session.run(model.y_predict, feed_dict=feed_dict);
        print("y_predict: " + str(result))

        feed_dict = {x:[[1., 5.8, 2.6,4., 1.2]]}
        result = session.run(model.logits, feed_dict=feed_dict);
        print("logits: " + str(result))
        result = session.run(model.y_predict, feed_dict=feed_dict);
        print("y_predict: " + str(result))

        feed_dict = {x:[[1., 7.1, 3., 5.9, 2.1]]}
        result = session.run(model.logits, feed_dict=feed_dict);
        print("logits: " + str(result))
        result = session.run(model.y_predict, feed_dict=feed_dict);
        print("y_predict: " + str(result))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        predict()
    else:
        train()

if __name__ == '__main__':
    main()
