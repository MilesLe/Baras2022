import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

D, tcga_maf, samples = pickle.load(open('/home/janaya2/Desktop/ATGC_paper/figures/tumor_classification/data/data.pkl', 'rb'))
tcga_maf = tcga_maf.loc[:, ['Tumor_Sample_Barcode', 'Hugo_Symbol', 'contexts']]

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
ncit_samples = samples.loc[samples['NCI-T Label'].isin(ncit_labels_kept)]
PCPG_ncit = ['Paraganglioma','Pheochromocytoma']
SARC_ncit = ['Desmoid-Type Fibromatosis','Leiomyosarcoma','Liposarcoma','Malignant Peripheral Nerve Sheath Tumor',
             'Myxofibrosarcoma','Synovial Sarcoma','Undifferentiated Pleomorphic Sarcoma']
TGCT_ncit = ['Testicular Non-Seminomatous Germ Cell Tumor','Testicular Seminoma']
ncit_samples.loc[ncit_samples['NCI-T Label'].isin(PCPG_ncit), 'NCI-T Label'] = 'PCPG'
ncit_samples.loc[ncit_samples['NCI-T Label'].isin(SARC_ncit), 'NCI-T Label'] = 'SARC'
ncit_samples.loc[ncit_samples['NCI-T Label'].isin(TGCT_ncit), 'NCI-T Label'] = 'TGCT'
#print(ncit_samples.loc[ncit_samples['NCI-T Label'] == 'Testicular Seminoma']['NCI-T Label'])
#print(list(set(ncit_samples['NCI-T Label'])))

A = ncit_samples['NCI-T Label'].astype('category')
samples = ncit_samples
###

context_df = tcga_maf.groupby(['Tumor_Sample_Barcode', "contexts"]).size().unstack(fill_value=0)
context_df = pd.DataFrame.from_dict({'Tumor_Sample_Barcode': context_df.index, 'context_counts': context_df.values.tolist()})
samples = pd.merge(samples, context_df, on='Tumor_Sample_Barcode')

gene_df = tcga_maf.groupby(['Tumor_Sample_Barcode', "Hugo_Symbol"]).size().unstack(fill_value=0)
gene_df = pd.DataFrame.from_dict({'Tumor_Sample_Barcode': gene_df.index, 'gene_counts': gene_df.values.tolist()})
samples = pd.merge(samples, gene_df, on='Tumor_Sample_Barcode')
samples['type'] = samples['type'].apply(lambda x: 'COAD' if x == 'READ' else x)

del D, tcga_maf

#A = samples['type'].astype('category')
classes = A.cat.categories.values
##integer values for random forest
y_label = np.arange(len(classes))[A.cat.codes]

class_counts = dict(zip(*np.unique(y_label, return_counts=True)))
y_weights = np.array([1 / class_counts[_] for _ in y_label])
y_weights /= np.sum(y_weights)
context_counts = np.stack(samples['context_counts'].values)
gene_counts = np.stack(samples['gene_counts'].values)

context_test_predictions = []
context_all_predictions = []
gene_test_predictions = []
gene_all_predictions = []
both_test_predictions = []
both_all_predictions = []
test_idx = []
cancer_aucs = []
reg = RandomForestClassifier(n_estimators=900, min_samples_split=10, random_state=0, n_jobs=20)
for idx_train, idx_test in StratifiedKFold(n_splits=5, random_state=0, shuffle=True).split(y_label, y_label):
    print('fold')
    test_idx.append(idx_test)
    y_train, y_test = y_label[idx_train], y_label[idx_test]
    ##for context counts
    #context_train, context_test = context_counts[idx_train], context_counts[idx_test]
    #reg.fit(context_train, y_train,
    #        sample_weight=y_weights[idx_train] / np.sum(y_weights[idx_train])
    #        )
    #context_test_predictions.append(reg.predict_proba(context_test))
    #context_all_predictions.append(reg.predict_proba(context_counts))

    ##for gene counts
    #gene_train, gene_test = gene_counts[idx_train], gene_counts[idx_test]
    #reg.fit(gene_train, y_train,
    #        sample_weight=y_weights[idx_train] / np.sum(y_weights[idx_train])
    #        )
    #gene_test_predictions.append(reg.predict_proba(gene_test))
    #gene_all_predictions.append(reg.predict_proba(gene_counts))

    #for gene and context
    both_train, both_test = np.concatenate([gene_counts[idx_train], context_counts[idx_train]], axis=-1), np.concatenate([gene_counts[idx_test], context_counts[idx_test]], axis=-1)
    both_all = np.concatenate([gene_counts, context_counts], axis=-1)
    reg.fit(both_train, y_train,
            sample_weight=y_weights[idx_train] / np.sum(y_weights[idx_train])
            )
    both_test_predictions.append(reg.predict_proba(both_test))
    both_all_predictions.append(reg.predict_proba(both_all))


predictions = both_test_predictions 

with open('/home/mlee276/Desktop/TCGA-ML-main/results/rf_both_predictions.pkl', 'wb') as f:
    pickle.dump([test_idx, predictions], f)

##weighted accuracy
print(np.sum((np.argmax(np.concatenate(predictions), axis=-1) == y_label[np.concatenate(test_idx)]) * y_weights[np.concatenate(test_idx)]))
##unweighted accuracy
print(sum(np.argmax(np.concatenate(predictions), axis=-1) == y_label[np.concatenate(test_idx)]) / len(y_label))
print(roc_auc_score(y_label[np.concatenate(test_idx)], np.concatenate(predictions), multi_class='ovr'))


##contexts
# 0.4747891674957764
# 0.49048022001269304
# 0.922272395013135

##genes
# 0.3948476786437132
# 0.474931246033425
# 0.8844072684036307

##contexts and genes
# 0.49995291772042944
# 0.575206261899725
# 0.9339863605603765
