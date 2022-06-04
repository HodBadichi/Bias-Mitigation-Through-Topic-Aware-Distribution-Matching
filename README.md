# Bias Mitigation Through Topic-Aware Distribution Matching - Final project :

This project consists with 2 parts, each contains its workflows:
1. Topic modeling part:
  * BERTtopic workflow (train the model, tune hyperparms, visualize and measure coherence)
  * LDA workflow (train the model, tune hyperparms, visualize and measure coherence)
 
 2. GAN
  * Frozen Bert workflow (train pre trained bert model over pubmed data)
  * Discriminator Workflow with Bert as embedder (using the DistributionMatching module - NoahArc)
  * Discriminator Workflow with Sentence bert as embedder (using the DistributionMatching module - NoahArc)
  * GAN Workflow (using the DistributionMatching module - NoahArc)
  
  
Each workflow has a "run" funcion that will envoke the full workflow,
where each workflow also responsible for getting the relevant data for the workflow and create the projects directory structure.

Each workflow has a "run_on_server.sh" file for running the workflow (the run function) as a batch job on Technion lambda server by running:
sbatch -c 2 --gres=gpu:1 run_on_server.sh -o run.out

Each workflow has a workflow_config file with the relevant Hyperparams for it



For example - running the BertTopic workflow:
1. cd TopicModeling\Bert\src
2. We will define the Hyperparams for bertopic training on hparams_config.py (add relevent min_topic_size_range, and n_gram_range)
3. In this dir, will run sbatch -c 2 --gres=gpu:1 run_on_server.sh -o run.out
4. on TopicModeling\Bert\saved_models will be our trained topic models
5. on TopicModeling\Bert\results will be the visualization of each topic model, coherence csv file for all models and a coherence graph visualization


Note - the project has a requirements file, run: 'pip install -r requirements.txt' before running the workflows.

Enjoy :)
