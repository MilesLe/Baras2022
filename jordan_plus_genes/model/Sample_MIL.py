import tensorflow as tf
from jordan_plus_genes.model.KerasLayers import Activations, Ragged, Embed, StrandWeight, Dropout

class InstanceModels:

    class GeneEmbed:
        def __init__(self, shape=None, dim=None, input_dim=None, regularization=.01):
            self.shape = shape
            self.regularization = regularization
            self.input_dim = input_dim
            self.dim = dim
            self.model = None
            self.build()

        def build(self, *args, **kwarg):
            input = tf.keras.layers.Input(self.shape, dtype=tf.int32)
            output = Embed(input_dimension=self.input_dim, embedding_dimension=self.dim, regularization=self.regularization, trainable=True)(input)
            ##we do a log on the graph so we can't have negative numbers
            output = tf.keras.activations.relu(output)
            self.model = tf.keras.Model(inputs=[input], outputs=[output])



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


class RaggedModels:
    class MIL:
        def __init__(self,
                     instance_encoders=[],
                     sample_encoders=[],
                     instance_layers=[],
                     output_dims=[1],
                     output_types=['classification'],
                     output_names=[],
                     regularization=.2,
                     mil_hidden=[32, 16],
                     attention_layers=[16],
                     dropout=0,
                     instance_dropout=0,
                     input_dropout=False):
            self.instance_encoders,\
            self.sample_encoders,\
            self.instance_layers,\
            self.output_dims,\
            self.output_types,\
            self.output_names,\
            self.regularization,\
            self.mil_hidden,\
            self.attention_layers,\
            self.dropout,\
            self.instance_dropout,\
            self.input_dropout = instance_encoders,\
                         sample_encoders,\
                         instance_layers,\
                         output_dims,\
                         output_types,\
                         output_names,\
                         regularization,\
                         mil_hidden,\
                         attention_layers,\
                         dropout,\
                         instance_dropout,\
                         input_dropout,\

            if self.output_names == []:
                self.output_names = ['output_' + str(index) for index, i in enumerate(self.output_types)]
            self.model, self.attention_model = None, None
            self.build()

        def build(self):
            ##get the instance inputs from the instance encoders
            ragged_inputs = [[tf.keras.layers.Input(shape=input_tensor.shape, dtype=input_tensor.dtype, ragged=True) for input_tensor in encoder.inputs] for encoder in self.instance_encoders]

            ##get the sample inputs from the sample encoders
            sample_inputs = [[tf.keras.layers.Input(shape=input_tensor.shape[1:], dtype=input_tensor.dtype) for input_tensor in encoder.inputs] for encoder in self.sample_encoders]

            ##if you had instance encoders do your instance encoding
            if self.instance_encoders != []:
                ragged_encodings = [Ragged.MapFlatValues(encoder)(ragged_input) for ragged_input, encoder in zip(ragged_inputs, self.instance_encoders)]
                # flatten encoders if needed
                ragged_encodings = [Ragged.MapFlatValues(tf.keras.layers.Flatten())(ragged_encoding) for ragged_encoding in ragged_encodings]
                # concat different encodings
                ragged_fused = tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=2))(ragged_encodings)

                ##if you want a dropout prior to aggregation
                if self.instance_dropout:
                    ragged_hidden = [Ragged.MapFlatValues(tf.keras.layers.Dropout(self.instance_dropout))(ragged_fused)]
                else:
                    ragged_hidden = [ragged_fused]

                ##if you want additional layers prior to aggregation
                for i in self.instance_layers:
                    ragged_hidden.append(Ragged.MapFlatValues(tf.keras.layers.Dense(units=i, activation=tf.keras.activations.relu))(ragged_hidden[-1]))
                    if self.dropout:
                        ragged_hidden.append(Ragged.MapFlatValues(tf.keras.layers.Dropout(self.dropout))(ragged_hidden[-1]))

                ##aggregation
                pooling, ragged_attention_weights = Ragged.Attention(regularization=self.regularization, layers=self.attention_layers)(ragged_hidden[-1])
                pooled_hidden = [pooling[:, 0, :]]

                #######from here on we at the sample level

                ##if you did data dropout have to account for that here during evaluation
                if self.input_dropout:
                    pooled_hidden = [Dropout(self.input_dropout)(pooled_hidden[-1])]

                ##sums can get large if there are a lot of mutations so we'll do a log
                pooled_hidden = [tf.math.log(pooled_hidden[-1] + 1)]

            ##if you have sample data do the encodings here
            if self.sample_encoders != []:
                sample_encodings = [encoder(sample_input) for sample_input, encoder in zip(sample_inputs, self.sample_encoders)]
                sample_fused = tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=-1))(sample_encodings)
                if self.instance_encoders != []:
                    fused = [tf.concat([pooled_hidden[-1], sample_fused], axis=-1)]
                else:
                    fused = [sample_fused]
            else:
                fused = [pooled_hidden[-1]]

            ##perform your final layers
            for i in self.mil_hidden:
                fused.append(tf.keras.layers.Dense(units=i, activation='relu')(fused[-1]))
                if self.dropout:
                    fused.append(tf.keras.layers.Dropout(self.dropout)(fused[-1]))

            ##make your outputs (we only have one but you could have more)
            output_tensors = []
            for output_type, output_dim, output_name in zip(self.output_types, self.output_dims, self.output_names):
                output_tensors.append(tf.keras.layers.Dense(units=output_dim, activation=None, name=output_name)(fused[-1]))

            ##the main model
            self.model = tf.keras.Model(inputs=ragged_inputs + sample_inputs, outputs=output_tensors)
            ##a model to get attention weights
            if self.instance_encoders != []:
                self.attention_model = tf.keras.Model(inputs=ragged_inputs + sample_inputs, outputs=[ragged_attention_weights])