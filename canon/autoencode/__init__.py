import tensorflow as tf
import keras.backend as K
import horovod.keras as hvd

from .models import AE_128_to_256, AE_128_to_64

# !!! remember to clear session/graph if you rebuild your graph to avoid out-of-memory errors !!!
def reset_tf_session(nersc=False):
	K.clear_session()
    tf.reset_default_graph()
	if nersc:
		# Horovod: initialize Horovod.
		hvd.init()
		# Horovod: pin GPU to be used to process local rank (one GPU per process)
		config=tf.ConfigProto(inter_op_parallelism_threads=int(os.environ['NUM_INTER_THREADS']),
    					intra_op_parallelism_threads=int(os.environ['NUM_INTRA_THREADS']))
		config.gpu_options.allow_growth = True
		config.gpu_options.visible_device_list = str(hvd.local_rank())
		K.set_session(tf.Session(config=config))
    
    s = K.get_session()
    return s
