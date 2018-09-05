from keras.layers import Conv2D, Dense, MaxPooling2D, AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from utils.keras_sparisity_regularization import SparsityRegularization
from copy import deepcopy
import numpy as np


def freeze_SR_layer(model, prune_rate=0.):
    for layer in model.layers:
        if isinstance(layer, SparsityRegularization):
            w = layer.get_weights()
            nw = deepcopy(w[0])
            ind = len(nw) - int(len(nw) * (1 - prune_rate))
            nw = np.sort(np.abs(nw), axis=-1)
            threshold = nw[ind]
            w[0][np.abs(w[0]) < threshold] = 0
            layer.set_weights(w)
            layer.trainable = False


def set_compact_model_weights(origin_model, compact_model):
    weights = {}
    # todo: support BN and GlobalAveragePooling2D
    for o, c in zip(origin_model.layers, compact_model.layers):
        if o.input_shape == c.input_shape and o.output_shape == c.output_shape:
            w = o.get_weights()
            if len(w) > 0:
                weights[c.name] = w

        if isinstance(o, SparsityRegularization):
            w = o.get_weights()
            # get not zero value index
            idx = np.argwhere(w[0] != 0)
            k = [n[0] for n in idx]
            # set compact SR layer weights
            nw = deepcopy(w)
            nw[0] = nw[0][k,]
            c.set_weights(nw)
            input_name = o.input.name

            for j in range(len(origin_model.layers)):
                o_layer = origin_model.layers[j]
                c_layer = compact_model.layers[j]
                if o_layer.output.name == input_name:
                    if isinstance(o_layer, Conv2D):
                        cw = o_layer.get_weights()
                        ncw = deepcopy(cw[0])
                        k = [n[0] for n in idx]
                        ncw = ncw[:, :, :, k]
                        if o_layer.use_bias:
                            ncb = deepcopy(cw[1])
                            ncb = ncb[k,]
                            weights[c_layer.name] = [ncw, ncb]
                        else:
                            weights[c_layer.name] = [ncw]
                        continue
                    else:
                        input_name = o_layer.input.name

    for i in range(len(origin_model.layers)):
        o = origin_model.layers[i]
        if isinstance(o, SparsityRegularization):
            w = o.get_weights()
            # get not zero value index
            idx = np.argwhere(w[0] != 0)
            k = [n[0] for n in idx]
            output_name = o.output.name
            # process succeeding convolution or dense layer
            pool_output_shape = None
            for j in range(len(origin_model.layers)):
                o_layer = origin_model.layers[j]
                c_layer = compact_model.layers[j]
                if o_layer.input.name == output_name:
                    if isinstance(o_layer, Conv2D):
                        if c_layer.name in weights:
                            cw = weights[c_layer.name]
                        else:
                            cw = o_layer.get_weights()
                        ncw = deepcopy(cw[0])
                        k = [n[0] for n in idx]
                        ncw = ncw[:, :, k, :]
                        if o_layer.use_bias:
                            weights[c_layer.name] = [ncw, cw[1]]
                        else:
                            weights[c_layer.name] = [ncw]
                        continue
                    elif isinstance(o_layer, Dense):
                        cw = o_layer.get_weights()
                        tmp = deepcopy(cw[0])
                        tmp = np.reshape(tmp, (pool_output_shape[1], pool_output_shape[2],
                                               pool_output_shape[3], tmp.shape[-1]))
                        tmp = tmp[:, :, k, :]
                        tmp = np.reshape(tmp, (np.prod(tmp.shape[0:-1]), tmp.shape[-1]))
                        weights[c_layer.name] = [tmp, cw[1]]
                        continue
                    elif isinstance(o_layer, MaxPooling2D):
                        pool_output_shape = o_layer.output_shape
                        output_name = o_layer.output.name
                    elif isinstance(o_layer, AveragePooling2D):
                        pool_output_shape = o_layer.output_shape
                        output_name = o_layer.output.name
                    else:
                        output_name = o_layer.output.name

    for layer in compact_model.layers:
        if layer.name in weights:
            layer.set_weights(weights[layer.name])
