# Baras2022
Data files needed: 
- /home/janaya2/Desktop/ATGC_paper/figures/tumor_classification/data/data.pkl
- cancerhotspots.v2.maf
- cases.2020-02-28.json and mc3.v0.2.8.PUBLIC.maf and 96c file -  but this was for 1st sem stuff so maybe don't include
- publication_hotspots.vcf 
- *deepgene stuff*

Steps:
1. Create a results folder to hold the predictions and weights of the models.
2. Change the file paths to the correct path 
3. Run all the models on all forms of data appropriate.
4. Run the "analysis" and "attention" notebooks
5. Run deep gene stuff

Note:
- Deepgene LR and RF figures and data provided by Curtis


predictions and weights: 


This repository contains the code and data sources created and used by Miles Lee during the 2022 Spring semester in Dr. Alexander Baras' lab. Here are instructions to load the code/data and create the figures in the report. 

Load data:
1. Download the raw data from https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/mlee276_jh_edu/EvJK2lp3CnhCiXo7pnIeomQBaD5LUQtAsxztVO3jGF7UAg?e=IEnfSi
2. Download the random forest, neural network, and MIL model predictions and weights from https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/mlee276_jh_edu/Ehb2IvYAwXxPmIFhiUZQwd8B3y-sltEH1iT7b23yCv2efA?e=GbAcg8 

Load code:
1. Download this repository.
2. Change the file paths to be correct. Below are the file paths in the code and their associated file that you will download.
    - /home/sahn33/Documents/cancerhotspots.v2.maf <= data/cancerhotspots.v2.maf
    - publication_hotspots.vcf <= data/publication_hotspots.vcf
3. Run the ___ programs to train and save the random forests, neural network, and MIL models for both the "gene" and "context" data.
   Here are the saved models (predictions and weights):
    - /home/janaya2/Desktop/ATGC_paper/figures/tumor_classification/data/data.pkl <= data/data.pkl
    - /home/mlee276/Desktop/TCGA-ML-main/results/mil_gene_predictions.pkl <= results/mil_gene_predictions.pkl
    - /home/mlee276/Desktop/TCGA-ML-main/results/nn_gene_predictions.pkl <= results/nn_gene_predictions.pkl
    - /home/mlee276/Desktop/TCGA-ML-main/results/rf_gene_predictions.pkl <= results/rf_gene_predictions.pkl
    - /home/mlee276/Desktop/TCGA-ML-main/results/mil_contexts_predictions.pkl <= results/mil_contexts_predictions.pkl
    - /home/mlee276/Desktop/TCGA-ML-main/results/nn_contexts_predictions.pkl <= results/nn_contexts_predictions.pkl
    - /home/mlee276/Desktop/TCGA-ML-main/results/rf_contexts_predictions.pkl <= results/rf_contexts_predictions.pkl
    - /home/mlee276/Desktop/TCGA-ML-main/results/nn_both_predictions.pkl <= results/nn_both_predictions.pkl
    - /home/mlee276/Desktop/TCGA-ML-main/results/rf_both_predictions.pkl <= results/rf_both_predictions.pkl

Run code to produce figures:
- 

