# Baras2022
Note:
- Deepgene LR and RF figures and data provided by Curtis

This repository contains the code and data sources created and used by Miles Lee during the 2022 Spring semester in Dr. Alexander Baras' lab. Here are instructions to load the code/data and create the figures in the report. 

Load data:
1. Download the raw data from https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/mlee276_jh_edu/EvJK2lp3CnhCiXo7pnIeomQBaD5LUQtAsxztVO3jGF7UAg?e=IEnfSi
2. Download the random forest, neural network, and MIL model predictions and weights from https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/mlee276_jh_edu/Ehb2IvYAwXxPmIFhiUZQwd8B3y-sltEH1iT7b23yCv2efA?e=GbAcg8 

Load code:
1. Download this repository.
2. Change the file paths to be correct. Below are the file paths in the code and their associated file that you will download.
    - /home/sahn33/Documents/cancerhotspots.v2.maf <= data/cancerhotspots.v2.maf
    - publication_hotspots.vcf <= data/publication_hotspots.vcf
3. Run the run_mil_context.py, run_mil_gene.py, run_nn.py, and run_rf.py programs to train and save the random forests, neural network, and MIL models for both the "gene" and "context" data. run_nn.py and run_rf.py have commented code that needs to be caredully uncommented to run the model on the gene, context, and gene & context data respectively. 
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
- analysis-context, analysis-gene, and analysis-gene-context: the figures are produced by calling functions. Some functions produce figures that visualize one model and data combination (ex. random forest on gene data). Therefore, some functions will need to have their input changed to visualize different models and data. 
- attention-context and attention-gene: 

