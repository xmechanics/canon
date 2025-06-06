# import os
# import tensorflow as tf
# import keras.backend as K


# # !!! remember to clear session/graph if you rebuild your graph to avoid out-of-memory errors !!!
# def reset_tf_session(nersc=False):
#     K.clear_session()
#     tf.reset_default_graph()
#     if nersc:
#         config = tf.ConfigProto(inter_op_parallelism_threads=int(os.environ['NUM_INTER_THREADS']),
#                                 intra_op_parallelism_threads=int(os.environ['NUM_INTRA_THREADS']))
#         K.set_session(tf.Session(config=config))
#     s = K.get_session()
#     return s


import tensorflow as tf
from tensorflow import keras     # guarantees you’re on tf.keras
from tensorflow.keras import backend as K

def reset_tf_session(nersc=False):
    K.clear_session()            # clears residual models/graphs

    if nersc:                    # optional CPU thread pinning
        import os, tensorflow as tf
        tf.config.threading.set_inter_op_parallelism_threads(
            int(os.environ['NUM_INTER_THREADS']))
        tf.config.threading.set_intra_op_parallelism_threads(
            int(os.environ['NUM_INTRA_THREADS']))