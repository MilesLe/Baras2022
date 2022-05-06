import numpy as np
import pandas as pd
import json
from jordan.model.Sample_MIL import RaggedModels, InstanceModels
from jordan.model.KerasLayers import Losses, Metrics
import tensorflow as tf
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from jordan.model import DatasetsUtils
import pickle

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[-4], True)
tf.config.experimental.set_visible_devices(physical_devices[-4], 'GPU')

D, tcga_maf, samples = pickle.load(open('/home/janaya2/Desktop/ATGC_paper/figures/tumor_classification/data/data.pkl', 'rb'))
del tcga_maf

strand_emb_mat = np.concatenate([np.zeros(2)[np.newaxis, :], np.diag(np.ones(2))], axis=0)
D['strand_emb'] = strand_emb_mat[D['strand']]

###
# filtering the NCI-T labels (https://livejohnshopkins-my.sharepoint.com/:x:/r/personal/abaras1_jh_edu/_layouts/15/doc2.aspx?sourcedoc=%7B5f92f0fc-ec6c-40d5-ab17-0d3345f9f2c2%7D&action=edit&activeCell=%27Sheet1%27!B21&wdinitialsession=e072a38f-57c8-4c1f-885b-efaefcc81d35&wdrldsc=2&wdrldc=1&wdrldr=AccessTokenExpiredWarning%2CRefreshingExpiredAccessT)
ncit_labels_kept = ['Muscle-Invasive Bladder Carcinoma','Infiltrating Ductal Breast Carcinoma',
                    'Invasive Lobular Breast Carcinoma','Cervical Squamous Cell Carcinoma',
                    'Colorectal Adenocarcinoma','Glioblastoma','Head and Neck Squamous Cell Carcinoma',
                    'Clear Cell Renal Cell Carcinoma','Papillary Renal Cell Carcinoma','Astrocytoma',
                    'Oligoastrocytoma','Oligodendroglioma','Hepatocellular Carcinoma','Lung Adenocarcinoma',
                    'Lung Squamous Cell Carcinoma','Ovarian Serous Adenocarcinoma','Adenocarcinoma, Pancreas',
                    'Paraganglioma','Pheochromocytoma','Prostate Acinar Adenocarcinoma','Colorectal Adenocarcinoma',
                    'Desmoid-Type Fibromatosis','Leiomyosarcoma','Liposarcoma','Malignant Peripheral Nerve Sheath Tumor',
                    'Myxofibrosarcoma','Synovial Sarcoma','Undifferentiated Pleomorphic Sarcoma',
                    'Cutaneous Melanoma','Gastric Adenocarcinoma','Testicular Non-Seminomatous Germ Cell Tumor',
                    'Testicular Seminoma','Thyroid Gland Follicular Carcinoma','Thyroid Gland Papillary Carcinoma',
                    'Endometrial Endometrioid Adenocarcinoma','Endometrial Serous Adenocarcinoma']
idx_filter = samples['NCI-T Label'].isin(ncit_labels_kept)
ncit_samples = samples.loc[idx_filter]
PCPG_ncit = ['Paraganglioma','Pheochromocytoma']
SARC_ncit = ['Desmoid-Type Fibromatosis','Leiomyosarcoma','Liposarcoma','Malignant Peripheral Nerve Sheath Tumor',
             'Myxofibrosarcoma','Synovial Sarcoma','Undifferentiated Pleomorphic Sarcoma']
TGCT_ncit = ['Testicular Non-Seminomatous Germ Cell Tumor','Testicular Seminoma']
ncit_samples.loc[ncit_samples['NCI-T Label'].isin(PCPG_ncit), 'NCI-T Label'] = 'PCPG'
ncit_samples.loc[ncit_samples['NCI-T Label'].isin(SARC_ncit), 'NCI-T Label'] = 'SARC'
ncit_samples.loc[ncit_samples['NCI-T Label'].isin(TGCT_ncit), 'NCI-T Label'] = 'TGCT'
#print(ncit_samples.loc[ncit_samples['NCI-T Label'] == 'Testicular Seminoma']['NCI-T Label'])
#print(list(set(ncit_samples['NCI-T Label'])))

#samples = ncit_samples
A = ncit_samples['NCI-T Label'].astype('category')
###

##each instance has an associated sample index so we need to group the instances for each sample together
##the sample dataframe had its index reset so the indexes start at 0 and are concurrent, if you subset the sample dataframe then you'll have to only generate indexes for those samples
indexes = [np.where(D['sample_idx'] == idx) for idx in np.arange(samples.shape[0])[idx_filter]]

##the sequence encoder has five inputs
five_p = np.array([D['seq_5p'][i] for i in indexes], dtype='object')
three_p = np.array([D['seq_3p'][i] for i in indexes], dtype='object')
ref = np.array([D['seq_ref'][i] for i in indexes], dtype='object')
alt = np.array([D['seq_alt'][i] for i in indexes], dtype='object')
strand = np.array([D['strand_emb'][i] for i in indexes], dtype='object')

##we want to dropout instances at the level of the batch
dropout = .4
index_loader = DatasetsUtils.Map.FromNumpytoIndices([j for i in indexes for j in i], dropout=dropout)

five_p_loader = DatasetsUtils.Map.FromNumpyandIndices(five_p, tf.int16)
three_p_loader = DatasetsUtils.Map.FromNumpyandIndices(three_p, tf.int16)
ref_loader = DatasetsUtils.Map.FromNumpyandIndices(ref, tf.int16)
alt_loader = DatasetsUtils.Map.FromNumpyandIndices(alt, tf.int16)
strand_loader = DatasetsUtils.Map.FromNumpyandIndices(strand, tf.float32)

##when we evaluate the model we don't want to be dropping out any instances
five_p_loader_eval = DatasetsUtils.Map.FromNumpy(five_p, tf.int16)
three_p_loader_eval = DatasetsUtils.Map.FromNumpy(three_p, tf.int16)
ref_loader_eval = DatasetsUtils.Map.FromNumpy(ref, tf.int16)
alt_loader_eval = DatasetsUtils.Map.FromNumpy(alt, tf.int16)
strand_loader_eval = DatasetsUtils.Map.FromNumpy(strand, tf.float32)

#A = samples['type'].astype('category')
classes = A.cat.categories.values
##integer values for random forest
classes_onehot = np.eye(len(classes))[A.cat.codes]
y_label = classes_onehot

y_strat = np.argmax(y_label, axis=-1)
class_counts = dict(zip(*np.unique(y_strat, return_counts=True)))
y_weights = np.array([1 / class_counts[_] for _ in y_strat])
y_weights /= np.sum(y_weights)

y_label_loader = DatasetsUtils.Map.FromNumpy(y_label, tf.float32)
y_weights_loader = DatasetsUtils.Map.FromNumpy(y_weights, tf.float32)


test_idx = []
weights = []
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_weighted_CE', min_delta=0.001, patience=100, mode='min', restore_best_weights=True)]
for idx_train, idx_test in StratifiedKFold(n_splits=5, random_state=0, shuffle=True).split(y_strat, y_strat):
    eval = 100
    test_idx.append(idx_test)
    idx_train, idx_valid = [idx_train[idx] for idx in list(StratifiedShuffleSplit(n_splits=1, test_size=1000, random_state=0).split(np.zeros_like(y_strat)[idx_train], y_strat[idx_train]))[0]]
    with tf.device('/cpu:0'):
        ds_train = tf.data.Dataset.from_tensor_slices((idx_train, y_strat[idx_train]))
        ds_train = ds_train.apply(DatasetsUtils.Apply.StratifiedMinibatch(batch_size=512, ds_size=len(idx_train)))

        ds_train = ds_train.map(lambda x: ((
            index_loader(x),
        )

        ),
                                )

        ds_train = ds_train.map(lambda x: ((
                                                five_p_loader(x[0], x[1]),
                                                three_p_loader(x[0], x[1]),
                                                ref_loader(x[0], x[1]),
                                                alt_loader(x[0], x[1]),
                                                strand_loader(x[0], x[1]),
                                               ),
                                              (
                                                  y_label_loader(x[0]),
                                              ),
                                               y_weights_loader(x[0])
        ),
                                )

        ds_train.prefetch(1)
        ds_valid = tf.data.Dataset.from_tensor_slices(((
                                               five_p_loader_eval(idx_valid),
                                               three_p_loader_eval(idx_valid),
                                               ref_loader_eval(idx_valid),
                                               alt_loader_eval(idx_valid),
                                               strand_loader_eval(idx_valid),
                                           ),
                                            (
                                                tf.gather(y_label, idx_valid),
                                            ),
                                            tf.gather(y_weights, idx_valid)
                                            ))
        ds_valid = ds_valid.batch(len(idx_valid), drop_remainder=False)


    losses = [Losses.CrossEntropy()]
    for i in range(3):
        sequence_encoder = InstanceModels.VariantSequence(6, 4, 2, [16, 16, 16, 16], fusion_dimension=256)
        mil = RaggedModels.MIL(instance_encoders=[sequence_encoder.model], sample_encoders=[], output_dims=[y_label.shape[-1]], output_types=['other'], instance_layers=[512], mil_hidden=[256, 128, 64], attention_layers=[64, 16], dropout=.2, instance_dropout=.3, regularization=0, input_dropout=dropout)

        mil.model.compile(loss=losses,
                          metrics=[Metrics.CrossEntropy(), Metrics.Accuracy()],
                          weighted_metrics=[Metrics.CrossEntropy()],
                          optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,
                                                             ))

        mil.model.fit(ds_train,
                      steps_per_epoch=20,
                      epochs=20000,
                      validation_data=ds_valid,
                      callbacks=callbacks,
                      )
        run_eval = mil.model.evaluate(ds_valid)[-1]

        if run_eval < eval:
            eval = run_eval
            run_weights = mil.model.get_weights()

    weights.append(run_weights)


with open('/home/mlee276/Desktop/TCGA-ML-main/jordan/mil_encoder/contexts_weights.pkl', 'wb') as f:
    pickle.dump([test_idx, weights], f)


predictions = []
for idx_test, weight in zip(test_idx, weights):
    mil.model.set_weights(weight)
    ds_test = tf.data.Dataset.from_tensor_slices(((
                                                       five_p_loader_eval(idx_test),
                                                       three_p_loader_eval(idx_test),
                                                       ref_loader_eval(idx_test),
                                                       alt_loader_eval(idx_test),
                                                       strand_loader_eval(idx_test),
                                                   ),
                                                   (
                                                       tf.gather(y_label, idx_test),
                                                   ),
                                                   tf.gather(y_weights, idx_test)
    ))
    ds_test = ds_test.batch(len(idx_valid), drop_remainder=False)
    predictions.append(mil.model.predict(ds_test))


P = np.concatenate(predictions)
#convert the model logits to probablities
z = np.exp(P - np.max(P, axis=1, keepdims=True))
predictions = z / np.sum(z, axis=1, keepdims=True)


#with open('/home/janaya2/Desktop/ATGC_paper/figures/tumor_classification/mil_encoder/results/contexts_predictions.pkl', 'wb') as f:
#    pickle.dump([test_idx, predictions], f)
with open('/home/mlee276/Desktop/TCGA-ML-main/jordan/mil_encoder/contexts_predictions.pkl', 'wb') as f:
    pickle.dump([test_idx, predictions], f)

print(np.sum((np.argmax(predictions, axis=-1) == np.argmax(y_label[np.concatenate(test_idx)], axis=-1)) * y_weights[np.concatenate(test_idx)]))
print(sum(np.argmax(predictions, axis=-1) == np.argmax(y_label[np.concatenate(test_idx)], axis=-1)) / len(y_label))
print(roc_auc_score(np.argmax(y_label[np.concatenate(test_idx)], axis=-1), predictions, multi_class='ovr'))

# 0.5437881229146698
# 0.5161836259784218
# 0.9446450201200127