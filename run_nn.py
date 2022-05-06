from re import M
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
tcga_maf = tcga_maf.loc[:, ['Tumor_Sample_Barcode', 'contexts', 'Hugo_Symbol']]

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

context_df = tcga_maf.groupby(['Tumor_Sample_Barcode', "contexts"]).size().unstack(fill_value=0)
context_df = pd.DataFrame.from_dict({'Tumor_Sample_Barcode': context_df.index, 'context_counts': context_df.values.tolist()})
samples = pd.merge(samples, context_df, on='Tumor_Sample_Barcode')

gene_df = tcga_maf.groupby(['Tumor_Sample_Barcode', "Hugo_Symbol"]).size().unstack(fill_value=0)
gene_df = pd.DataFrame.from_dict({'Tumor_Sample_Barcode': gene_df.index, 'gene_counts': gene_df.values.tolist()})
samples = pd.merge(samples, gene_df, on='Tumor_Sample_Barcode')
samples['type'] = samples['type'].apply(lambda x: 'COAD' if x == 'READ' else x)

#A = samples['type'].astype('category') #<-- uncomment for non NCIT labels
classes = A.cat.categories.values
##integer values for random forest
classes_onehot = np.eye(len(classes))[A.cat.codes]
y_label = classes_onehot

y_strat = np.argmax(y_label, axis=-1)
class_counts = dict(zip(*np.unique(y_strat, return_counts=True)))
y_weights = np.array([1 / class_counts[_] for _ in y_strat])
y_weights /= np.sum(y_weights)
context_counts = np.apply_along_axis(lambda x: np.log(x + 1), 0, np.stack(samples['context_counts'].values))[idx_filter]
gene_counts = np.apply_along_axis(lambda x: np.log(x + 1), 0, np.stack(samples['gene_counts'].values))[idx_filter]
both_counts = np.concatenate([context_counts, gene_counts], axis=-1)

predictions = []
evaluations = []
test_idx = []
weights = []
batch_size = 512
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=25, mode='min', restore_best_weights=True)]
for idx_train, idx_test in StratifiedKFold(n_splits=5, random_state=0, shuffle=True).split(y_strat, y_strat):
    test_idx.append(idx_test)
    temp_evaluations = []
    eval = 100
    idx_train, idx_valid = [idx_train[idx] for idx in list(StratifiedShuffleSplit(n_splits=1, test_size=1000, random_state=0).split(np.zeros_like(y_strat)[idx_train], y_strat[idx_train]))[0]]

    ds_train = tf.data.Dataset.from_tensor_slices((idx_train, y_strat[idx_train]))
    ds_train = ds_train.apply(DatasetsUtils.Apply.StratifiedMinibatch(batch_size=batch_size, ds_size=len(idx_train)))
    ds_train = ds_train.map(lambda x: ((
                                          #tf.gather(context_counts, x),
                                          tf.gather(gene_counts, x)
                                          #tf.gather(both_counts, x)
                                           ),
                                          (
                                          tf.gather(y_label, x),
                                          ),
                                           tf.gather(y_weights, x)
                                          )
                            )

    ds_valid = tf.data.Dataset.from_tensor_slices((
                                                  (
                                                   #context_counts[idx_valid],
                                                    gene_counts[idx_valid],
                                                   #both_counts[idx_valid]
                                                   ),
                                                  (
                                                   y_label[idx_valid],
                                                  ),
                                                   y_weights[idx_valid]
                                                   ))
    ds_valid = ds_valid.batch(len(idx_valid), drop_remainder=False)

    ds_test = tf.data.Dataset.from_tensor_slices((
                                                 (
                                                  #context_counts[idx_test],
                                                   gene_counts[idx_test],
                                                  #both_counts[idx_test]
                                                 ),
                                                 (
                                                  y_label[idx_test],
                                                 ),

                                                  y_weights[idx_test]
                                                  ))

    ds_test = ds_test.batch(len(idx_test), drop_remainder=False)

    losses = [Losses.CrossEntropy()]

    for run in range(3):

        # context and genes
        #both_encoder = InstanceModels.Feature(shape=(both_counts.shape[-1]), input_dropout=.5, layer_dropouts=[.5], layers=[1024, 512])
        #mil = RaggedModels.MIL(instance_encoders=[], sample_encoders=[both_encoder.model], output_dims=[y_label.shape[-1]], output_types=['other'], mil_hidden=[256], dropout=.5)

        #contexts
        #context_encoder = InstanceModels.Feature(shape=(context_counts.shape[-1]), input_dropout=.1, layers=[])
        #mil = RaggedModels.MIL(instance_encoders=[], sample_encoders=[context_encoder.model], output_dims=[y_label.shape[-1]], output_types=['other'], mil_hidden=[512, 256, 128], dropout=.3)

        ##genes
        gene_encoder = InstanceModels.Feature(shape=(gene_counts.shape[-1]), input_dropout=.5, layer_dropouts=[.5], layers=[1052])
        mil = RaggedModels.MIL(instance_encoders=[], sample_encoders=[gene_encoder.model], output_dims=[y_label.shape[-1]], output_types=['other'], mil_hidden=[512], dropout=.5)
        #
        mil.model.compile(loss=losses,
                          metrics=[Metrics.CrossEntropy(), Metrics.Accuracy()],
                          optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,
                                                             ))
        mil.model.fit(ds_train,
                      steps_per_epoch=10,
                      epochs=20000,
                      validation_data=ds_valid,
                      callbacks=callbacks)
        run_eval = mil.model.evaluate(ds_valid)[0]
        temp_evaluations.append(run_eval)
        if run_eval < eval:
            eval = run_eval
            run_weights = mil.model.get_weights()
            print('test_eval', mil.model.evaluate(ds_test))
    mil.model.set_weights(run_weights)
    predictions.append(mil.model.predict(ds_test))
    weights.append(run_weights)
    evaluations.append(temp_evaluations)

with open('/home/mlee276/Desktop/TCGA-ML-main/results/nn_gene_weights.pkl', 'wb') as f:
    pickle.dump([test_idx, weights], f)

P = np.concatenate(predictions)
#convert the model logits to probablities
z = np.exp(P - np.max(P, axis=1, keepdims=True))
predictions = z / np.sum(z, axis=1, keepdims=True)

matrix = confusion_matrix(np.argmax(y_label[np.concatenate(test_idx)], axis=-1), np.argmax(predictions, axis=-1))
print(matrix)

with open('/home/mlee276/Desktop/TCGA-ML-main/results/nn_gene_predictions.pkl', 'wb') as f:
    pickle.dump([test_idx, predictions], f)

print(np.sum((np.argmax(predictions, axis=-1) == np.argmax(y_label[np.concatenate(test_idx)], axis=-1)) * y_weights[np.concatenate(test_idx)]))
print(sum(np.argmax(predictions, axis=-1) == np.argmax(y_label[np.concatenate(test_idx)], axis=-1)) / len(y_label))
print(roc_auc_score(np.argmax(y_label[np.concatenate(test_idx)], axis=-1), predictions, multi_class='ovr'))




# contexts
# context_encoder = SampleModels.Feature(shape=(context_counts.shape[-1]), input_dropout=.1, layer_dropout=.2, layers=[])
# mil = RaggedModels.MIL(instance_encoders=[], sample_encoders=[context_encoder.model], output_dims=[y_label.shape[-1]], output_types=['other'], mil_hidden=[512, 256, 128], mode='none', dropout=.3)
#sums
# 0.49005842612678463
# 0.4775756293632325
# 0.9261276179946987

##genes
# gene_encoder = InstanceModels.Feature(shape=(gene_counts.shape[-1]), input_dropout=.5, layer_dropouts=[.5], layers=[1052])
# mil = RaggedModels.MIL(instance_encoders=[], sample_encoders=[gene_encoder.model], output_dims=[y_label.shape[-1]], output_types=['other'], mil_hidden=[512], mode='none', dropout=.5)

# 0.4685102393262379
# 0.5289824412946901
# 0.9084559789238952

##contexts and genes
# both_encoder = InstanceModels.Feature(shape=(both_counts.shape[-1]), input_dropout=.5, layer_dropouts=[.5], layers=[1024, 512])
# mil = RaggedModels.MIL(instance_encoders=[], sample_encoders=[both_encoder.model], output_dims=[y_label.shape[-1]], output_types=['other'], mil_hidden=[256], mode='none', dropout=.5)
# 0.5823694613500944
# 0.6223820605034905
# 0.943716020658751

