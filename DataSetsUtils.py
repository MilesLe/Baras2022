import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold


class Apply:

    class StratifiedMinibatch:
        def __init__(self, batch_size, ds_size, reshuffle_each_iteration=True):
            self.batch_size, self.ds_size, self.reshuffle_each_iteration = batch_size, ds_size, reshuffle_each_iteration
            # max number of splits
            self.n_splits = (self.ds_size // self.batch_size) + 1
            # stratified "mini-batch" via k-fold
            if self.n_splits > 1:
                self.batcher = StratifiedKFold(n_splits=self.n_splits, shuffle=self.reshuffle_each_iteration)
            else:
                self.batcher = None

        def __call__(self, ds_input: tf.data.Dataset):
            def generator():
                # expecting ds of (idx, y_strat)
                idx, y_strat = list(map(tf.stack, list(map(list, zip(*list(ds_input))))))
                while True:
                    if self.batcher is not None:
                        for _, batch_idx in self.batcher.split(y_strat, y_strat):
                            yield tf.gather(idx, batch_idx, axis=0)
                    else:
                        yield idx

            return tf.data.Dataset.from_generator(generator, output_types=ds_input.element_spec[0].dtype, output_shapes=(None, ))

    class StratifiedBootstrap:
        def __init__(self, batch_class_sizes):
            self.batch_class_sizes = batch_class_sizes
            self.batch_size = sum(self.batch_class_sizes)
            self.rnd = tf.random.Generator.from_non_deterministic_state()

        def __call__(self, ds_input: tf.data.Dataset):
            def generator():
                # expecting ds of (idx, y_true, y_strat)
                idx, y_true, y_strat = list(map(tf.stack, list(map(list, zip(*list(ds_input))))))
                assert (tf.reduce_max(y_strat).numpy() + 1) == len(self.batch_class_sizes)
                class_idx = [tf.where(y_strat == i)[:, 0] for i in range(len(self.batch_class_sizes))]
                while True:
                    batch_idx = list()
                    for j in range(len(self.batch_class_sizes)):
                        batch_idx.append(tf.gather(class_idx[j], self.rnd.uniform(shape=(self.batch_class_sizes[j], ), maxval=tf.cast(class_idx[j].shape[0], tf.int64), dtype=tf.int64)))
                    batch_idx = tf.concat(batch_idx, axis=0)

                    yield tf.gather(idx, batch_idx, axis=0), tf.gather(y_true, batch_idx, axis=0)

            return tf.data.Dataset.from_generator(generator,
                                                  output_types=(ds_input.element_spec[0].dtype, ds_input.element_spec[1].dtype),
                                                  output_shapes=((self.batch_size, ), (self.batch_size, ds_input.element_spec[1].shape[0])))


class Map:

    class LoadBatchByIndices:
        def loader(self):
            raise NotImplementedError

        def __call__(self, sample_idx):
            # flat_values and additional_args together should be the input into the ragged_constructor of the loader
            flat_values, *additional_args = tf.py_function(self.loader, [sample_idx], self.tf_output_types)
            flat_values.set_shape((None,) + self.inner_shape)

            return self.ragged_constructor(flat_values, *additional_args) if self.ragged_constructor is not None else flat_values

    class FromNumpy(LoadBatchByIndices):
        def __init__(self, data, instance_dropout=None):
            self.data = data
            self.instance_dropout = instance_dropout
            if self.data.dtype == np.dtype('O'):
                shapes = list(set(list(map(lambda x: x.shape[1:], self.data))))
                dtypes = list(set(list(map(lambda x: x.dtype, self.data))))
                assert len(shapes) == 1 and len(dtypes) == 1
                self.inner_shape = shapes[0]
                self.tf_output_types = [tf.dtypes.as_dtype(dtypes[0]), tf.int32]
                self.ragged_constructor = tf.RaggedTensor.from_row_lengths
            else:
                self.inner_shape = data.shape[1:]
                self.tf_output_types = [tf.dtypes.as_dtype(self.data.dtype), tf.int32]
                self.ragged_constructor = None
            
        def loader(self, indices):
            if self.data.dtype == np.dtype('O'):
                batch = list()
                for idx in indices.numpy():
                    if self.instance_dropout is None:
                        batch.append(self.data[idx])
                    else:
                        batch.append(self.data[idx][np.random.uniform(0, 1, self.data[idx].shape[0]) < self.instance_dropout])
                return np.concatenate(batch, axis=0), np.array([v.shape[0] for v in batch])
            else:
                idx = indices.numpy()
                return self.data[idx], np.ones_like(idx)
