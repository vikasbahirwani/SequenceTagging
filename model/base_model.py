import os
import tensorflow as tf
import numpy as np

class BaseModel(object):
    """ Generic class containing methods not specific to NER """

    def __int__(self, config):
        """Initializes logger and config, as well as declares session and saver for instances of this class

        :param config: instance of Config class containing hyper parameters, vocab, embeddings etc.
        """

        self.config = config
        self.logger = config.logger
        self.session = None
        self.saver = None

    def reinitialize_weights(self, scope_name):
        """Re-initializes weights under the given scope name

        :param scope_name: weights (aka tf variables) from this scope will be reinitialized
        """

        variables = tf.contrib.framework.get_variables(scope_name)
        init = tf.variables_initializer(variables)
        self.session.run(init)

    def add_train_op(self, lr_method,lr, loss, clip = -1):
        """Defines the self_train_op that performs updates on a batch

        :param lr_method: (string, lower-case) sgd method. Valid values: "adam", "adagrad", "sgd", "rmsprop"
        :param lr: (tf.placeholder) tf.float32, learning rate
        :param loss: (tensor) tf.float32 loss to minimize
        :param clip: (python float) clipping of gradient. If < 0, no clipping
        """
        lr_m = lr_method.lower() # lowering case for good measure

        with tf.variable_scope("train_step"):
            if lr_m == 'adam':
                optimizer = tf.train.AdamOptimizer(lr)
            elif lr_m == 'adagrad':
                optimizer = tf.train.AdagradDAOptimizer(lr)
            elif lr_m == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lr)
            elif lr_m == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(lr)
            else:
                raise NotImplementedError("Unsupported lr_method supplied {:}".format(lr_method))

            # TODO: Review the following:
            if clip > 0:  # gradient clipping if clip is positive
                grads, vs = zip(*optimizer.compute_gradients(loss))
                grads, _ = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(loss)

    def initialize_session(self):
        """
        Initializes session and saver for this instance, and initializes all the variables.
        """

        self.logger.info("Initializing a new tf session")
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def restore_session(self, dir_model):
        """
        Restore the model weights from the latest checkpoint

        :param dir_model: model directory where the checkpoints are stored
        """

        self.logger.info("Restoring weights from the latest checkpoint")
        self.saver.restore(self.session, dir_model)

    def save_session(self):
        """
        Saves the weights in the current session to dir_model defined in config
        """

        self.logger.info("Saving weights from the current session")

        if not os.path.isdir(self.config.dir_model):
            os.mkdir(self.config.dir_model)

        self.saver.Save(self.session, self.config.dir_model)

    def close_session(self):
        """
        Closes the current session
        """

        self.session.close();

    def add_summary(self):
        """
        Add variables for tensorboard (merged_all and file writer)
        """

        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.dir_output, self.session.graph)

    def train(self, train, dev):
        """
        Trains the model with early stopping and exponential lr decay
        TODO: add/update the format info for arguments

        :param train: training set of the form (sentence,tag)
        :param dev: dev set
        """

        best_score = 0
        nepochs_no_improv = 0 # for early stopping

        self.add_summary() # getting tensorboard ready

        for epoch in np.arange(self.config.nepochs):
            self.logger.info("Epoch {:} of {:}".format(epoch, self.config.nepochs))

            score = self.run_epoch()
            self.lr *= self.config.lr_decay # exponential decay

            # early stopping
            if score > best_score:
                best_score = score
                self.save_session()
                self.logger.info(" - new best score {:}".format(best_score))
                nepochs_no_improv = 0
            else:
                nepochs_no_improv += 1

                if nepochs_no_improv >= self.confing.nepochs_no_improv:
                    self.logger.info(" - early stopping because of {:} epochs without improvement".format(nepochs_no_improv))
                    break

    def run_epoch(self, train, dev, epoch):
        """
        Empty method that the child class must implement.
        This method will use the train and dev sets to run the epoch
        :param train: training set
        :param dev: dev set
        :param epoch: epoch number
        :return score using dev set
        """

        raise Exception("Not implemented, child class must implement")

    def evaluate(self, test):
        """
        Evaluates the model on the test set
        TODO: add/update the format info for arguments

        :param test: test set
        """

        self.logger.info("Evaluating the model on the test set")
        metrics = self.run_evaluate(test)
        msg = " - ".join(["{} {:04.2f}".format(k, v) for k, v in metrics.items()])
        self.logger.info(msg)

    def run_evaluate(self,test):
        """
        Empty method that the child class must implement.
        This method will use the test set to evaluate the model.
        :param test: test set
        :return metrics
        """

        raise Exception("Not implemented, child class must implement")