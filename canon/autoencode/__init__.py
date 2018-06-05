import tensorflow as tf
import keras.backend as K

from .models import AE_128_to_256, AE_128_to_64

# !!! remember to clear session/graph if you rebuild your graph to avoid out-of-memory errors !!!
def reset_tf_session(nersc=False):
    K.clear_session()
    tf.reset_default_graph()
    if nersc:
    	s = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=int(os.environ['NUM_INTER_THREADS']),
    					intra_op_parallelism_threads=int(os.environ['NUM_INTRA_THREADS'])))
    else:
    	s = K.get_session()
    return s
