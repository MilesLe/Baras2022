import tensorflow as tf
from jordan.model.KerasLayers import Activations, Ragged, Embed, StrandWeight, Dropout

class InstanceModels:
    
    #class Gene_Embed

    class VariantPositionBin:
        def __init__(self, bins, fusion_dimension=128, default_activation=tf.keras.activations.relu):
            self.bins = bins
            self.fusion_dimension = fusion_dimension
            self.default_activation = default_activation
            self.model = None
            self.build()

        def build(self, *args, **kwargs):
            bin_inputs = [tf.keras.layers.Input(shape=(), dtype=tf.int32) for i in self.bins]
            pos_input = tf.keras.layers.Input(shape=(1,), dtype=tf.float32)
            embeds = [Embed(embedding_dimension=i, trainable=False) for i in self.bins]
            bins_fused = tf.concat([emb(i) for i, emb in zip(bin_inputs, embeds)], axis=-1)
            bins_fused = tf.keras.layers.Dense(units=sum(self.bins), activation=Activations.ARU())(bins_fused)
            pos_loc = tf.keras.layers.Dense(units=64, activation=Activations.ASU())(pos_input)
            pos_loc = tf.keras.layers.Dense(units=32, activation=Activations.ARU())(pos_loc)
            fused = tf.concat([bins_fused, pos_loc], axis=-1)
            fused = tf.keras.layers.Dense(units=self.fusion_dimension, activation=Activations.ARU())(fused)
            self.model = tf.keras.Model(inputs=bin_inputs + [pos_input], outputs=[fused])


    class VariantBin:
        def __init__(self, bins, layers=[], default_activation=tf.keras.activations.relu, fused_dropout=0, layer_dropout=0):
            self.bins = bins
            self.layers = layers
            self.default_activation = default_activation
            self.layer_dropout = layer_dropout
            self.fused_dropout = fused_dropout
            self.model = None
            self.build()

        def build(self, *args, **kwargs):
            bin_inputs = [tf.keras.layers.Input(shape=(), dtype=tf.int32) for i in self.bins]
            embeds = [Embed(embedding_dimension=i, trainable=False) for i in self.bins]
            bins_fused = [tf.concat([emb(i) for i, emb in zip(bin_inputs, embeds)], axis=-1)]
            if self.fused_dropout:
                bins_fused.append(tf.keras.layers.Dropout(self.fused_dropout)(bins_fused[-1]))
            for index, i in enumerate(self.layers):
                bins_fused.append(tf.keras.layers.Dense(units=i, activation=self.default_activation)(bins_fused[-1]))
                bins_fused.append(tf.keras.layers.Dropout(self.layer_dropout)(bins_fused[-1]))
            self.model = tf.keras.Model(inputs=bin_inputs, outputs=[bins_fused[-1]])


    class VariantSequence:
        def __init__(self, sequence_length, sequence_embedding_dimension, n_strands, convolution_params, fusion_dimension=64, default_activation=tf.keras.activations.relu, use_frame=False, regularization=.01):
            self.sequence_length = sequence_length
            self.sequence_embedding_dimension = sequence_embedding_dimension
            self.convolution_params = convolution_params
            self.default_activation = default_activation
            self.n_strands = n_strands
            self.use_frame = use_frame
            self.fusion_dimension = fusion_dimension
            self.regularization=regularization
            self.model = None
            self.build()

        def build(self, *args, **kwargs):
            five_p = tf.keras.layers.Input(shape=(self.sequence_length, self.n_strands), dtype=tf.int32)
            three_p = tf.keras.layers.Input(shape=(self.sequence_length, self.n_strands), dtype=tf.int32)
            ref = tf.keras.layers.Input(shape=(self.sequence_length, self.n_strands), dtype=tf.int32)
            alt = tf.keras.layers.Input(shape=(self.sequence_length, self.n_strands), dtype=tf.int32)
            strand = tf.keras.layers.Input(shape=(self.n_strands,), dtype=tf.float32)

            # layers of convolution for sequence feature extraction based on conv_params
            features = [[]] * 4
            convolutions = [[]] * 4
            nucleotide_emb = Embed(embedding_dimension=4, trainable=False)
            for index, feature in enumerate([five_p, three_p, ref, alt]):
                convolutions[index] = tf.keras.layers.Conv2D(filters=self.convolution_params[index], kernel_size=[1, self.sequence_length], activation=Activations.ARU())
                # apply conv to forward and reverse
                features[index] = tf.stack([convolutions[index](nucleotide_emb(feature)[:, tf.newaxis, :, i, :]) for i in range(self.n_strands)], axis=3)
                # pool over any remaining positions
                features[index] = tf.reduce_max(features[index], axis=[1, 2])

            fused = tf.concat(features, axis=2)
            fused = tf.keras.layers.Dense(units=self.fusion_dimension, activation=self.default_activation, kernel_regularizer=tf.keras.regularizers.l2(self.regularization))(fused)
            fused = tf.reduce_max(StrandWeight(trainable=True, n_features=fused.shape[2])(strand) * fused, axis=1)

            if self.use_frame:
                cds = tf.keras.layers.Input(shape=(3,), dtype=tf.float32)
                frame = tf.concat([strand, cds], axis=-1)
                frame = tf.keras.layers.Dense(units=6, activation=self.default_activation)(frame)
                fused = tf.concat([fused, frame], axis=-1)
                self.model = tf.keras.Model(inputs=[five_p, three_p, ref, alt, strand, cds], outputs=[fused])
            else:
                self.model = tf.keras.Model(inputs=[five_p, three_p, ref, alt, strand], outputs=[fused])

    class PassThrough:
        def __init__(self, shape=None):
            self.shape = shape
            self.model = None
            self.build()

        def build(self, *args, **kwarg):
            input = tf.keras.layers.Input(self.shape, dtype=tf.float32)
            self.model = tf.keras.Model(inputs=[input], outputs=[input])

    class Feature:
        def __init__(self, shape=None, input_dropout=.5, layer_dropouts=[.2, .2], layers=[64, 32]):
            self.shape = shape
            self.model = None
            self.input_dropout = input_dropout
            self.layer_dropouts = layer_dropouts
            self.layers = layers
            self.build()

        def build(self, *args, **kwarg):
            input = tf.keras.layers.Input(self.shape, dtype=tf.float32)
            hidden = [tf.keras.layers.Dropout(self.input_dropout)(input)]
            for i, j in zip(self.layers, self.layer_dropouts):
                hidden.append(tf.keras.layers.Dense(units=i, activation=tf.keras.activations.relu)(hidden[-1]))
                hidden.append(tf.keras.layers.Dropout(j)(hidden[-1]))
            self.model = tf.keras.Model(inputs=[input], outputs=[hidden[-1]])

    class Type:
        def __init__(self, shape=None, dim=None):
            self.shape = shape
            self.dim = dim
            self.model = None
            self.build()

        def build(self, *args, **kwarg):
            input = tf.keras.layers.Input(self.shape, dtype=tf.int32)
            # type_emb = Embed(embedding_dimension=self.dim, trainable=False)
            # self.model = tf.keras.Model(inputs=[input], outputs=[type_emb(input)])
            self.model = tf.keras.Model(inputs=[input], outputs=[tf.one_hot(input, self.dim)])

    class Reads:
        def __init__(self, read_layers=[8, 16], fused_layers=[32, 64]):
            self.read_layers = read_layers
            self.fused_layers = fused_layers
            self.model = None
            self.build()

        def build(self, *args, **kwargs):
            ref_input = tf.keras.layers.Input(shape=(1,), dtype=tf.float32)
            ref = [ref_input]
            for i in self.read_layers:
                ref.append(tf.keras.layers.Dense(units=i, activation='relu')(ref[-1]))
            alt_input = tf.keras.layers.Input(shape=(1,), dtype=tf.float32)
            alt = [alt_input]
            for i in self.read_layers:
                alt.append(tf.keras.layers.Dense(units=i, activation='relu')(alt[-1]))
            fused = [tf.concat([ref[-1], alt[-1]], axis=-1)]
            for i in self.fused_layers:
                fused.append(tf.keras.layers.Dense(units=i, activation='relu')(fused[-1]))

            self.model = tf.keras.Model(inputs=[ref_input, alt_input], outputs=[fused[-1]])


class RaggedModels:
    class MIL:
        def __init__(self,
                     instance_encoders=[],
                     sample_encoders=[],
                     instance_layers=[],
                     sample_layers=[],
                     pooled_layers=[],
                     output_dims=[1],
                     output_types=['classification'],
                     output_names=[],
                     mode='attention',
                     pooling='sum',
                     regularization=.2,
                     fusion='after',
                     mil_hidden=[32, 16],
                     dynamic_hidden=[64, 32],
                     attention_layers=[16],
                     dropout=0,
                     instance_dropout=0,
                     input_dropout=False,
                     heads=1):
            self.instance_encoders,\
            self.sample_encoders,\
            self.instance_layers,\
            self.sample_layers,\
            self.pooled_layers,\
            self.output_dims,\
            self.output_types,\
            self.output_names,\
            self.mode,\
            self.pooling,\
            self.regularization,\
            self.fusion,\
            self.mil_hidden,\
            self.dynamic_hidden,\
            self.attention_layers,\
            self.dropout,\
            self.instance_dropout,\
            self.input_dropout,\
            self.heads = instance_encoders,\
                         sample_encoders,\
                         instance_layers,\
                         sample_layers,\
                         pooled_layers,\
                         output_dims,\
                         output_types,\
                         output_names,\
                         mode,\
                         pooling,\
                         regularization,\
                         fusion,\
                         mil_hidden,\
                         dynamic_hidden,\
                         attention_layers,\
                         dropout,\
                         instance_dropout,\
                         input_dropout,\
                         heads

            if self.output_names == []:
                self.output_names = ['output_' + str(index) for index, i in enumerate(self.output_types)]
            self.model, self.attention_model = None, None
            self.build()


        def build(self):
            ragged_inputs = [[tf.keras.layers.Input(shape=input_tensor.shape, dtype=input_tensor.dtype, ragged=True) for input_tensor in encoder.inputs] for encoder in self.instance_encoders]
            sample_inputs = [[tf.keras.layers.Input(shape=input_tensor.shape[1:], dtype=input_tensor.dtype) for input_tensor in encoder.inputs] for encoder in self.sample_encoders]

            ##sample level model encodings
            if self.sample_encoders != []:
                sample_encodings = [encoder(sample_input) for sample_input, encoder in zip(sample_inputs, self.sample_encoders)]
                sample_fused = tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=-1))(sample_encodings)

            if self.instance_encoders != []:
                ragged_encodings = [Ragged.MapFlatValues(encoder)(ragged_input) for ragged_input, encoder in zip(ragged_inputs, self.instance_encoders)]
                # flatten encoders if needed
                ragged_encodings = [Ragged.MapFlatValues(tf.keras.layers.Flatten())(ragged_encoding) for ragged_encoding in ragged_encodings]

                # based on the design of the input and graph instances can be fused prior to bag aggregation
                ragged_fused = tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=2))(ragged_encodings)

                if self.instance_dropout:
                    ragged_fused = Ragged.MapFlatValues(tf.keras.layers.Dropout(self.instance_dropout))(ragged_fused)

                if self.sample_encoders != []:
                    if self.fusion == 'before':
                        ragged_hidden = [Ragged.Dense(units=64, activation=tf.keras.activations.relu)((ragged_fused, sample_fused))]
                    else:
                        ragged_hidden = [ragged_fused]
                else:
                    ragged_hidden = [ragged_fused]

                for i in self.instance_layers:
                    ragged_hidden.append(Ragged.MapFlatValues(tf.keras.layers.Dense(units=i, activation=tf.keras.activations.relu))(ragged_hidden[-1]))
                    if self.dropout:
                        ragged_hidden.append(Ragged.MapFlatValues(tf.keras.layers.Dropout(self.dropout))(ragged_hidden[-1]))

                if self.mode == 'attention':
                    if self.pooling == 'both':
                        pooling, ragged_attention_weights = Ragged.Attention(pooling='mean', regularization=self.regularization, layers=self.attention_layers)(ragged_hidden[-1])
                        pooled_hidden = [tf.concat([pooling[:, 0, :], tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(ragged_attention_weights)], axis=-1)]
                    elif self.pooling == 'dynamic':
                        pooling_1, ragged_attention_weights_1 = Ragged.Attention(pooling='mean', regularization=self.regularization, layers=self.attention_layers)(ragged_hidden[-1])
                        for index, i in enumerate(self.dynamic_hidden):
                            if index == 0:
                                instance_ragged_fused = [Ragged.Dense(units=i, activation=tf.keras.activations.relu)((ragged_hidden[-1], pooling_1[:, 0, :]))]
                            else:
                                instance_ragged_fused.append(Ragged.MapFlatValues(tf.keras.layers.Dense(units=i, activation=tf.keras.activations.relu))(instance_ragged_fused[-1]))
                        pooling_2, ragged_attention_weights = Ragged.Attention(pooling='dynamic', regularization=self.regularization, layers=self.attention_layers)([ragged_hidden[-1], instance_ragged_fused[-1]])
                        pooled_hidden = [tf.math.log(pooling_2[:, 0, :] + 1)]
                    else:
                        pooling, ragged_attention_weights = Ragged.Attention(pooling=self.pooling, regularization=self.regularization, layers=self.attention_layers)(ragged_hidden[-1])
                        pooled_hidden = [tf.math.log(pooling[:, 0, :] + 1)]

                else:
                    if self.pooling == 'mean':
                        pooling = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=ragged_hidden[-1].ragged_rank))(ragged_hidden[-1])
                    else:
                        pooling = tf.math.log(tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=ragged_hidden[-1].ragged_rank))(ragged_hidden[-1]) + 1)
                    pooled_hidden = [pooling]

                if self.pooling != 'mean' and self.input_dropout:
                    pooled_hidden = [Dropout(self.input_dropout)(pooled_hidden[-1])]

                for i in self.pooled_layers:
                    pooled_hidden.append(tf.keras.layers.Dense(units=i, activation=tf.keras.activations.relu)(pooled_hidden[-1]))

            if self.sample_encoders != []:
                if self.fusion == 'after':
                    if self.instance_encoders != []:
                        fused = [tf.concat([pooled_hidden[-1], sample_fused], axis=-1)]
                    else:
                        fused = [sample_fused]
                else:
                    fused = [pooled_hidden[-1]]

            else:
                fused = [pooled_hidden[-1]]

            for i in self.sample_layers:
                fused.append(tf.keras.layers.Dense(units=i, activation=tf.keras.activations.relu)(fused[-1]))

            for i in self.mil_hidden:
                fused.append(tf.keras.layers.Dense(units=i, activation='relu')(fused[-1]))
                if self.dropout:
                    fused.append(tf.keras.layers.Dropout(self.dropout)(fused[-1]))

            output_tensors = []
            for output_type, output_dim, output_name in zip(self.output_types, self.output_dims, self.output_names):
                if output_type == 'quantiles':
                    output_layers = (8, 1)
                    point_estimate, lower_bound, upper_bound = list(), list(), list()
                    for i in range(len(output_layers)):
                        point_estimate.append(tf.keras.layers.Dense(units=output_layers[i], activation=None if i == (len(output_layers) - 1) else tf.keras.activations.softplus)(fused[-1] if i == 0 else point_estimate[-1]))

                    for l in [lower_bound, upper_bound]:
                        for i in range(len(output_layers)):
                            l.append(tf.keras.layers.Dense(units=output_layers[i], activation=tf.keras.activations.softplus)(fused[-1] if i == 0 else l[-1]))

                    output_tensors.append(tf.keras.activations.softplus(tf.concat([point_estimate[-1] - lower_bound[-1], point_estimate[-1], point_estimate[-1] + upper_bound[-1]], axis=1, name=output_name)))

                elif output_type == 'survival':
                    output_layers = (8, 4, 1)
                    pred = list()
                    for i in range(len(output_layers)):
                        pred.append(tf.keras.layers.Dense(units=output_layers[i], activation=None if i == (len(output_layers) - 1) else tf.keras.activations.relu, name=None if i == (len(output_layers) - 1) else output_name)(fused[-1] if i == 0 else pred[-1]))

                    output_tensors.append(pred[-1])

                elif output_type == 'regression':
                    ##assumes log transformed output
                    pred = tf.keras.layers.Dense(units=output_dim, activation='softplus', name=output_name)(fused[-1])
                    output_tensors.append(tf.math.log(pred + 1))

                elif output_type == 'anlulogits':
                    output_tensors.append(tf.keras.layers.Dense(units=output_dim, activation=Activations.ARU(), name=output_name)(fused[-1]))

                elif output_type == 'classification_probability':
                    probabilities = tf.keras.layers.Dense(units=output_dim, activation=Activations.ARU())(fused[-1])
                    probabilities = probabilities / tf.expand_dims(tf.reduce_sum(probabilities, axis=-1), axis=-1, name=output_name)
                    output_tensors.append(probabilities)

                else:
                    output_tensors.append(tf.keras.layers.Dense(units=output_dim, activation=None, name=output_name)(fused[-1]))

            self.model = tf.keras.Model(inputs=ragged_inputs + sample_inputs, outputs=output_tensors)

            if self.mode == 'attention':
                self.attention_model = tf.keras.Model(inputs=ragged_inputs + sample_inputs, outputs=[ragged_attention_weights])

    class MIL_heads:
        def __init__(self,
                     instance_encoders=[],
                     sample_encoders=[],
                     instance_layers=[],
                     sample_layers=[],
                     pooled_layers=[],
                     output_dims=[1],
                     output_types=['classification'],
                     output_names=[],
                     mode='attention',
                     pooling='sum',
                     regularization=.2,
                     fusion='after',
                     mil_hidden=[32, 16],
                     dynamic_hidden=[64, 32],
                     attention_layers=[16],
                     dropout=0,
                     instance_dropout=0,
                     input_dropout=0,
                     heads=1):
            self.instance_encoders,\
            self.sample_encoders,\
            self.instance_layers,\
            self.sample_layers,\
            self.pooled_layers,\
            self.output_dims,\
            self.output_types,\
            self.output_names,\
            self.mode,\
            self.pooling,\
            self.regularization,\
            self.fusion,\
            self.mil_hidden,\
            self.dynamic_hidden,\
            self.attention_layers,\
            self.dropout,\
            self.instance_dropout,\
            self.input_dropout,\
            self.heads = instance_encoders,\
                         sample_encoders,\
                         instance_layers,\
                         sample_layers,\
                         pooled_layers,\
                         output_dims,\
                         output_types,\
                         output_names,\
                         mode,\
                         pooling,\
                         regularization,\
                         fusion,\
                         mil_hidden,\
                         dynamic_hidden,\
                         attention_layers,\
                         dropout,\
                         instance_dropout,\
                         input_dropout,\
                         heads
            if self.output_names == []:
                self.output_names = ['output_' + str(index) for index, i in enumerate(self.output_types)]
            self.model, self.attention_model = None, None
            self.build()


        def build(self):
            ragged_inputs = [[tf.keras.layers.Input(shape=input_tensor.shape, dtype=input_tensor.dtype, ragged=True) for input_tensor in encoder.inputs] for encoder in self.instance_encoders]
            sample_inputs = [[tf.keras.layers.Input(shape=input_tensor.shape[1:], dtype=input_tensor.dtype) for input_tensor in encoder.inputs] for encoder in self.sample_encoders]

            ##sample level model encodings
            if self.sample_encoders != []:
                sample_encodings = [encoder(sample_input) for sample_input, encoder in zip(sample_inputs, self.sample_encoders)]
                sample_fused = tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=-1))(sample_encodings)

            ragged_encodings = [Ragged.MapFlatValues(encoder)(ragged_input) for ragged_input, encoder in zip(ragged_inputs, self.instance_encoders)]
            # flatten encoders if needed
            ragged_encodings = [Ragged.MapFlatValues(tf.keras.layers.Flatten())(ragged_encoding) for ragged_encoding in ragged_encodings]

            # based on the design of the input and graph instances can be fused prior to bag aggregation
            ragged_fused = tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=2))(ragged_encodings)

            if self.instance_dropout:
                ragged_fused = Ragged.MapFlatValues(tf.keras.layers.Dropout(self.instance_dropout))(ragged_fused)

            if self.sample_encoders != []:
                if self.fusion == 'before':
                    ragged_hidden = [Ragged.Dense(units=64, activation=tf.keras.activations.relu)((ragged_fused, sample_fused))]
                else:
                    ragged_hidden = [ragged_fused]
            else:
                ragged_hidden = [ragged_fused]

            for i in self.instance_layers:
                ragged_hidden.append(Ragged.MapFlatValues(tf.keras.layers.Dense(units=i, activation=tf.keras.activations.relu))(ragged_hidden[-1]))
                if self.dropout:
                    ragged_hidden.append(Ragged.MapFlatValues(tf.keras.layers.Dropout(self.dropout))(ragged_hidden[-1]))

            pooling, ragged_attention_weights = Ragged.Attention(pooling=self.pooling, regularization=self.regularization, layers=self.attention_layers, heads=self.heads)(ragged_hidden[-1])
            # attention_means = tf.keras.activations.softmax(tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1))(ragged_attention_weights))
            # attention_means = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1))(ragged_attention_weights)
            # attention_means = attention_means / tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(attention_means)



            if self.input_dropout:
                pooling = pooling * (1 - self.input_dropout)

            # hidden = tf.concat([[pooling[:, index, :]] for index in range(self.heads)], axis=-1)
            hidden = [tf.concat(tf.unstack(pooling, axis=1), axis=-1)]

            for i in self.mil_hidden:
                hidden.append(tf.keras.layers.Dense(units=i, activation='relu')(hidden[-1]))
                if self.dropout:
                    hidden.append(tf.keras.layers.Dropout(self.dropout)(hidden[-1]))
            hidden = tf.keras.layers.Dense(units=self.output_dims[0], activation=None)(hidden[-1])

            # pools = [[pooling[:, index, :]] for index in range(self.heads)]
            # for i in self.mil_hidden:
            #     for pool in pools:
            #         pool.append(tf.keras.layers.Dense(units=i, activation='relu')(pool[-1]))
            #         if self.dropout:
            #             pool.append(tf.keras.layers.Dropout(self.dropout)(pool[-1]))
            #
            # for pool in pools:
            #     pool.append(tf.keras.layers.Dense(units=self.output_dims[0], activation='softmax')(pool[-1]))
            #
            # pools = tf.stack([pool[-1] for pool in pools], axis=1)
            # pools = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(pools * attention_means[..., tf.newaxis])
            # pools = tf.keras.layers.Lambda(lambda x: tf.reduce_max(x, axis=1))(pools)



            self.model = tf.keras.Model(inputs=ragged_inputs + sample_inputs, outputs=[hidden])

            if self.mode == 'attention':
                self.attention_model = tf.keras.Model(inputs=ragged_inputs + sample_inputs, outputs=[ragged_attention_weights])