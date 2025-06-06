import math
import logging
import keras
from keras import layers, Model, backend
# try:
#     import keras_cv.models as cv_models
# except ImportError:
#     cv_models = None
cv_models = None

__logger = logging.getLogger(__name__)


def build(backbone_name: str,
          latent_features: int,
          input_shape: tuple[int, int, int] = (128,128,1)) -> tuple[Model, Model]:
    """
    backbone_name: "resnet50", "efficientnetb0", "convnext_tiny", or "densenet121"
    latent_features: size of the Dense bottleneck
    input_shape: e.g. (128,128,1)
    
    Recommended use cases for different backbone models:
    - ResNet50: Good balance between accuracy and speed, works well for general visual tasks
    - EfficientNetB0: Best accuracy per FLOP, ideal for resource-constrained environments
    - ConvNeXtTiny: Extra robustness with modern layers, better performance with fewer parameters
    - DenseNet121: Suitable for capturing small details like tiny defects
    
    Quick recommendations:
    - Need the smallest model that still "sees" tiny defects → ResNet-18 or DenseNet-121 (RadImageNet)
    - Want best accuracy per FLOP → EfficientNet-B0
    - Need extra robustness & modern layers → ConvNeXt-Tiny
    - Prefer an encoder that already sits inside an auto-encoder → ConvMAE
    
    Choose the backbone that best matches your compute budget and the size of your training set.
    This transfer-learned auto-encoder will converge much faster than training a symmetric CNN from scratch.
    """

    # 0) Figure out which backend we’re on
    bk = backend.backend()  # returns 'tensorflow', 'torch', or 'jax'
    is_torch = (bk == "torch")
    is_tf    = (bk == "tensorflow")
    # Log which backend we're using
    __logger.info(f"Using backend: {bk}")

    # 1) Input + gray→RGB hack
    inp = layers.Input(shape=input_shape, name="ae_input")
    x   = layers.Concatenate(name="to_rgb")([inp, inp, inp])

    # 2) Load the backbone (frozen)
    if backbone_name == "resnet50":
        if is_torch and cv_models:
            base = cv_models.ResNet50V2(
                include_top=False, weights="imagenet", input_tensor=x
            )
        elif is_tf:
            base = keras.applications.ResNet50V2(
                include_top=False, weights="imagenet", input_tensor=x
            )
        else:
            raise RuntimeError("ResNet50V2 requires keras-cv on PyTorch backend")
    elif backbone_name == "efficientnetb0":
        if is_torch and cv_models:
            base = cv_models.EfficientNetB0(
                include_top=False, weights="imagenet", input_tensor=x
            )
        elif is_tf:
            base = keras.applications.EfficientNetB0(
                include_top=False, weights="imagenet", input_tensor=x
            )
        else:
            raise RuntimeError("EfficientNetB0 requires keras-cv on PyTorch backend")
    elif backbone_name == "convnexttiny":
        if is_torch and cv_models:
            base = cv_models.ConvNeXtTiny(
                include_top=False, weights="imagenet", input_tensor=x
            )
        elif is_tf:
            base = keras.applications.ConvNeXtTiny(
                include_top=False, weights="imagenet", input_tensor=x
            )
        else:
            raise RuntimeError("ConvNeXtTiny requires keras-cv on PyTorch backend")
    elif backbone_name == "densenet121":
        if is_torch and cv_models:
            base = cv_models.DenseNet121(
                include_top=False, weights="imagenet", input_tensor=x
            )
        elif is_tf:
            base = keras.applications.DenseNet121(
                include_top=False, weights="imagenet", input_tensor=x
            )
        else:
            raise RuntimeError("DenseNet121 requires keras-cv on PyTorch backend")
    else:
        raise ValueError(f"Unknown backbone {backbone_name!r}")

    base.trainable = False

    # 3) Encoder head → latent
    h_enc, w_enc, c_enc = base.output_shape[1:]
    pooled = layers.GlobalAveragePooling2D(name="ae_gap")(base.output)
    latent = layers.Dense(latent_features,
                          activation="relu",
                          name="ae_latent")(pooled)
    encoder = Model(inp, latent, name=f"{backbone_name}_encoder")

    # 4) Decoder: latent → reshape → upsampling ladder
    dec_in = layers.Input(shape=(latent_features,), name="decoder_input")
    c_enc = max(c_enc, 2048)  # at least to 2048 channels
    flatten_size = h_enc * w_enc * c_enc
    x = layers.Dense(flatten_size,
                     activation="relu",
                     name="dec_dense")(dec_in)
    x = layers.Reshape((h_enc, w_enc, c_enc), name="dec_reshape")(x)

    up_factor = input_shape[0] // h_enc
    n_ups = int(math.log2(up_factor))
    filters = c_enc
    for i in range(n_ups):
        filters = max(filters // 2, 32)
        x = layers.Conv2DTranspose(filters,
                                   3,
                                   strides=2,
                                   padding="same",
                                   activation="relu",
                                   name=f"dec_tconv_{i}")(x)

    out = layers.Conv2DTranspose(1,
                                 3,
                                 padding="same",
                                 activation="sigmoid",
                                 name="dec_output")(x)
    decoder = Model(dec_in, out, name=f"{backbone_name}_decoder")

    return encoder, decoder


def compile_autoencoder(encoder, decoder):
    # grab the encoder’s Input() tensor and its Output()
    inp   = encoder.input   # e.g. a Tensor(shape=(None,128,128,1))
    latent = encoder.output

    # feed that latent directly into your decoder
    recon = decoder(latent)

    auto = Model(inp, recon, name="autoencoder")
    auto.compile(optimizer="adam",
                 loss="binary_crossentropy",
                 metrics=["mse"])
    return auto
