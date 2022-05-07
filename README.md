# Baras2022

This repository contains the code and data sources created and used by Miles Lee during the 2022 Spring semester in Dr. Alexander Baras' lab. Here are instructions to load the code/data and create the figures in the report. 

Note: many of the programs take a long time to run (at most an hour) and benefit from a lot of RAM.  

(A) Load data:
1. Download the raw data from https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/mlee276_jh_edu/EvJK2lp3CnhCiXo7pnIeomQBaD5LUQtAsxztVO3jGF7UAg?e=IEnfSi
2. Download the random forest, neural network, and MIL model predictions and weights from https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/mlee276_jh_edu/Ehb2IvYAwXxPmIFhiUZQwd8B3y-sltEH1iT7b23yCv2efA?e=GbAcg8 

(B) Load code:
1. Download this repository.
2. Change the file paths to be correct. Below are the file paths in the code and their associated file that you will download.
    - /home/sahn33/Documents/cancerhotspots.v2.maf <= data/cancerhotspots.v2.maf
    - publication_hotspots.vcf <= data/publication_hotspots.vcf
    - mc3.v0.2.8.PUBLIC.maf <= data/mc3.v0.2.8.PUBLIC.maf
    - WES_TCGA.96.csv <= data/WES_TCGA.96.csv
    - cases.2020-02-28.json <= data/cases.2020-02-28.json
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
4. Run the mc3_c96_NN notebook to see the results of my tensorflow feedforward simple neural network for the gene count and context data separately.
   Note: When switching between the gene and context data, make sure to assign the correct data to the input and label variables (x and y) and correct type of model (model for the context data and model_c96 for the gene data) during training.  

(C) Run code to produce figures:
- analysis-context, analysis-gene, and analysis-gene-context: the figures are produced by calling functions. Some functions produce figures that visualize one model and data combination (ex. random forest on gene data). Therefore, some functions will need to have their input changed to visualize different models and data. Follow the comments for details. 
- attention-context and attention-gene: Certian market cells (via a comment) produce visuals and must be ran. 

Credit:
Alexander Baras and Jordan Anaya produce the code in the jordan and jordan_plus_genes folder. Additionally, Jordan wrote part of the code in the run_mil_context.py, run_mil_gene.py, run_nn.py, and run_rf.py programs.
