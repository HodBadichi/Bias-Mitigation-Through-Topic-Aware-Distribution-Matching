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
it is missing anything, to start a workflow use "Run" function that invokes it.

In each folder where `run_on_server.sh` file exists  it should be used for running the workflow as a batch job on Technion 'lambda' server by using:
`sbatch -c 2 --gres=gpu:1 run_on_server.sh -o run.out` command. Use `hparams_config` file in each workflow to tune the hyperparams as desired

For example - BerTopic workflow:
1. Move to `TopicModeling\Bert\src`
2. Configure the Hyperparams for your BerTopic experiment in `TopicModeling\Bert\src\hparams_config.py` 
3. Run `sbatch -c 2 --gres=gpu:1 run_on_server.sh -o run.out` inside the lambda server
* Results :
4. trained topic models will be saved in `TopicModeling\Bert\saved_models`
5. visualizations and coherence csv files will be saved in  `TopicModeling\Bert\results`

Note - the project has a requirements file, run: `pip install -r requirements.txt` to create the environment

Enjoy :)
