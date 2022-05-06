import numpy as np
import pandas as pd
from jordan_plus_genes.model.Sample_MIL import InstanceModels, RaggedModels
from jordan_plus_genes.model.KerasLayers import Losses, Metrics
import tensorflow as tf
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from jordan.model import DatasetsUtils
import pickle

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[-2], True)
#tf.config.experimental.set_visible_devices(physical_devices[-2], 'GPU')

D, tcga_maf, samples = pickle.load(open('/home/janaya2/Desktop/ATGC_paper/figures/tumor_classification/data/data.pkl', 'rb'))
tcga_maf['Hugo_Symbol'] = tcga_maf['Hugo_Symbol'].astype('category')
samples['type'] = samples['type'].apply(lambda x: 'COAD' if x == 'READ' else x)

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

#samples = ncit_samples
A = ncit_samples['NCI-T Label'].astype('category')
###

D['genes'] = np.concatenate(tcga_maf[['Hugo_Symbol']].apply(lambda x: x.cat.codes).values + 1)
input_dim = max(D['genes'])
dropout = .5
indexes = [np.where(D['sample_idx'] == idx) for idx in np.arange(samples.shape[0])[idx_filter]]
#indexes = [np.where(D['sample_idx'] == idx) for idx in np.arange(samples.shape[0])]  #<-- uncomment for non NCIT labels
genes = np.array([D['genes'][i] for i in indexes], dtype='object')
index_loader = DatasetsUtils.Map.FromNumpytoIndices([j for i in indexes for j in i], dropout=dropout)
genes_loader = DatasetsUtils.Map.FromNumpyandIndices(genes, tf.int16)
genes_loader_eval = DatasetsUtils.Map.FromNumpy(genes, tf.int16, dropout=0)

#A = samples['type'].astype('category') #<-- uncomment for non NCIT labels
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

predictions = []
test_idx = []
weights = []
aucs = []
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_weighted_CE', min_delta=0.001, patience=30, mode='min', restore_best_weights=True)]
for idx_train, idx_test in StratifiedKFold(n_splits=5, random_state=0, shuffle=True).split(y_strat, y_strat):
    eval=100
    test_idx.append(idx_test)
    idx_train, idx_valid = [idx_train[idx] for idx in list(StratifiedShuffleSplit(n_splits=1, test_size=1000, random_state=0).split(np.zeros_like(y_strat)[idx_train], y_strat[idx_train]))[0]]
    with tf.device('/cpu:0'):
        ds_train = tf.data.Dataset.from_tensor_slices((idx_train, y_strat[idx_train]))
        ds_train = ds_train.apply(DatasetsUtils.Apply.StratifiedMinibatch(batch_size=512, ds_size=len(idx_train)))

        ds_train = ds_train.map(lambda x: ((index_loader(x),)),)

        ds_train = ds_train.map(lambda x: ((genes_loader(x[0], x[1]),), (y_label_loader(x[0]),), y_weights_loader(x[0])),)

        ds_train.prefetch(1)
        ds_valid = tf.data.Dataset.from_tensor_slices(((
                                               genes_loader_eval(idx_valid),
                                           ),
                                            (
                                                tf.gather(y_label, idx_valid),
                                            ),
                                            tf.gather(y_weights, idx_valid)
                                            ))
        ds_valid = ds_valid.batch(len(idx_valid), drop_remainder=False)


    losses = [Losses.CrossEntropy()]
    for i in range(3):
        gene_encoder = InstanceModels.GeneEmbed(shape=(), input_dim=input_dim, dim=256)
        mil = RaggedModels.MIL(instance_encoders=[gene_encoder.model], sample_encoders=[], output_dims=[y_label.shape[-1]], output_types=['other'], mil_hidden=[256], attention_layers=[64, 16], dropout=.5, instance_dropout=.5, regularization=0, input_dropout=dropout)

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

with open('/home/mlee276/Desktop/TCGA-ML-main/jordan_plus_genes/mil_encoder/gene_weights.pkl', 'wb') as f:
    pickle.dump([test_idx, weights], f)

for index, (idx_train, idx_test) in enumerate(StratifiedKFold(n_splits=5, random_state=0, shuffle=True).split(y_strat, y_strat)):
    mil.model.set_weights(weights[index])
    ds_test = tf.data.Dataset.from_tensor_slices(((
                                               genes_loader_eval(idx_test),
                                           ),
                                            (
                                                tf.gather(y_label, idx_test),
                                            ),
                                            tf.gather(y_weights, idx_test)
                                            ))
    ds_test = ds_test.batch(len(idx_test), drop_remainder=False)
    predictions.append(mil.model.predict(ds_test))
    mil.model.evaluate(ds_test)

P = np.concatenate(predictions)
#convert the model logits to probablities
z = np.exp(P - np.max(P, axis=1, keepdims=True))
predictions = z / np.sum(z, axis=1, keepdims=True)

with open('/home/mlee276/Desktop/TCGA-ML-main/jordan_plus_genes/mil_encoder/gene_predictions.pkl', 'wb') as f:
    pickle.dump([test_idx, predictions], f)

print(np.sum((np.argmax(predictions, axis=-1) == np.argmax(y_label[np.concatenate(test_idx)], axis=-1)) * y_weights[np.concatenate(test_idx)]))
print(sum(np.argmax(predictions, axis=-1) == np.argmax(y_label[np.concatenate(test_idx)], axis=-1)) / len(y_label))
print(roc_auc_score(np.argmax(y_label[np.concatenate(test_idx)], axis=-1), predictions, multi_class='ovr'))

# 0.47330276350160344
# 0.5259149566321134
# 0.9139275190070438

