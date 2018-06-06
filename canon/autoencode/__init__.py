import os
import tensorflow as tf
import keras.backend as K

from .builder import AE_128_to_256, AE_128_to_64


# !!! remember to clear session/graph if you rebuild your graph to avoid out-of-memory errors !!!
def reset_tf_session(nersc=False):
    K.clear_session()
    tf.reset_default_graph()
    if nersc:
        config = tf.ConfigProto(inter_op_parallelism_threads=int(os.environ['NUM_INTER_THREADS']),
                                intra_op_parallelism_threads=int(os.environ['NUM_INTRA_THREADS']))
        K.set_session(tf.Session(config=config))
    s = K.get_session()
    return s
