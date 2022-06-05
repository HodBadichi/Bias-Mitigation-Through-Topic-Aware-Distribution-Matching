# Bias Mitigation Through Topic-Aware Distribution Matching - project :

This project consists of 2 parts:
1. Topic modeling
  * BERTopic workflow, trains the model -> tunes hyperparams -> visualize and measure coherence
  * LDA workflow, trains the model -> tunes hyperparams -> visualize and measure coherence
 
 2. GAN 
  * FrozenBert workflow, trains pre-trained bert model over pubmed data
  * Discriminator workflow, using Bert as embedder
  * Discriminator workflow with Sentence bert as embedder 
  * GAN Workflow
  

Each workflow is treated as an independent script which generates the data and modify the project structure in case
it is missing anything, to make start a workflow use "Run" function that invokes it.

In each folder where `run_on_server.sh` file exists  it should be used for running the workflow (the run function) as a batch job on Technion 'lambda' server by using:
`sbatch -c 2 --gres=gpu:1 run_on_server.sh -o run.out`
Also, you can make use of `hparams_config` in each workflow to tune the hyperparams as you desire

For example - BertTopic workflow:
1. `cd TopicModeling\Bert\src`
2. We will define the Hyperparams for bertopic training in `hparams_config.py` 
3. Run `sbatch -c 2 --gres=gpu:1 run_on_server.sh -o run.out` inside the lambda server
* Results :
4. On `TopicModeling\Bert\saved_models` will consist our trained topic models
5. On `TopicModeling\Bert\results` will be the visualization of each topic model, coherence csv file for all models and a coherence graph visualization

Note - the project has a requirements file, run: `pip install -r requirements.txt` to create the environment

Enjoy :)
