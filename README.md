# Bias Mitigation Through Topic-Aware Distribution Matching - project:

## project structure
We assume the following tree structure for the project:
```
project
├── saved_models
└── Bias-Mitigation-Through-Topic-Aware-Distribution-Matching
```
This project consists of 2 parts:
1. Topic modeling - Mor and Hod project, 
  * BERTopic workflow, trains the model -> tunes hyperparams -> visualize and measure coherence
  * LDA workflow, trains the model -> tunes hyperparams -> visualize and measure coherence
 
 2. GAN  - Nofar and Liel + continuation of Mor and Hod's
  * FrozenBert workflow, trains pre-trained bert model over pubmed data
  * Discriminator workflow, using Bert as embedder
  * Discriminator workflow with Sentence bert as embedder with GPT-based and BERT-based models
  * GAN Workflow, using Bert as embedder
  * GAN Workflow, using GPT as embedder
  * GAN workflow, using BERT/GPT-based sentence transformers
  
Each workflow is treated as an independent script which generates the data and modifies the project structure in case it is missing anything, to start a workflow use "Run" function that invokes it.

## Running on CS servers
In each folder a `run_on_server.sh` file exists.
It should be used for running the workflow as a batch job on Technion 'lambda'/'newton' servers by typing:
`sbatch --gres=gpu:1 --nodelist={$NODE_NAME} run_on_server.sh` command.
The available nodes are different on each servers, to check which nodes are free to run on and what is their capcity, you can type `sgpu`

## Debugging on a specific node using VSCODE
We added debug configurations in `.vscode/launch.json`. 
In order to debug on a specific node, you can do the following:
1. Open bash on the node: `srun --gres=gpu:1 --nodelist={$NODE_NAME} --pty bash`
2. Type: `./code tunnel` and perform the neccasary authentication in github
3. Now you can open the IDE using the link, or directly in vscude (using the remote ssh extention for VScode):
    1. Type ctrl+shit+P and select Remote-Tunnels:Connect to tunnel using a github account
    2. select the tunnel that is available
4. Type ctrl+shift+D and select the desired debug configuration, debugging should start.

## Run configuration
Use `hparams_config` file in each workflow to tune the hyperparams as desired. 
WandB logging is used, use: `wandb login` before running the workflows. 

## Examples
PubMedGan workflow:
1. Move to `GAN/GANPubMed/src/`
2. Set the GAN to train at `PubMedGANworkflow.py`
   (e.g. `model = PubMedGanSentenceTransformer(hparams)`)
3. Configure the hyperparams for the GAN framework in `hparams_config.py`
3. Run on the newton/lambda server `sbatch --gres=gpu:1 --nodelist={$NODE_NAME} run_on_server.sh`
4. The logs of the run will be saved at `slurm-{RUN_ID}.out`
4. For sentence transformers: the model that is trained in the GAN will be saved in
   the previously mentioned `saved_models` folder 
   (e.g. at `project/saved_models/GAN_experiments_models/gpt2-medium_231114_145000.95985` - currently we save the model after each epoch but this can cause no space error)

BerTopic workflow:
1. Move to `TopicModeling\Bert\src`
2. Configure the Hyperparams for your BerTopic experiment in `TopicModeling\Bert\src\hparams_config.py` 
3. Run `sbatch --gres=gpu:1 run_on_server.sh` inside the lambda server
* Results :
4. trained topic models will be saved in `TopicModeling\Bert\saved_models`
5. visualizations and coherence csv files will be saved in  `TopicModeling\Bert\results`

## Enviornment
We suggest running the project on a virtual enviornment (venv) using conda.
1. Conda should be found under: {$USER HOMES FOLDER}/miniconda3
2. Create the venv using `conda create --name venv python=3.8.16`
3. Activate the venv using `source {$USER HOMES FOLDER}/miniconda3/bin/activate venv`

* We suggest adding the command to the .bashrc file so that this venv is activate with each new 
opened terminal

## Requirements
The project has two requirements files, run: 
1. `pip install -r old_requirements.txt` to create the environment neccassary for the generation of
the csvs and  similiarity+probaility matrices (Which need a specific version of BertTopic to work)
2. After you ran a discriminator/GAN workflow these files should be created under a `data` folder. These will be created once and then reused in future runs.
3. `pip install -r requirements.txt` to create the enviornment neccassary for running a GPT model in the GAN (which need a specific version of transformers, which sadly isn't compatible with the version BERTTopic uses)

## project drive
https://technionmail-my.sharepoint.com/:f:/g/personal/liel-blu_campus_technion_ac_il/EkgGgv7SvtpKpRg208l-W6EB89jl4w9lE99-Y0hXMYl1Yg?e=FmPSFk

In the drive you can find:
1. Data folder - contains the .csv and similiary+probability matrices neccasry for the project to run. can save time 
2. GAN models checkpoints- these are checkpoints where we (Nofar and Liel) finished, contains two models trained on medical data (one in the GAN using discriminator and one without)
3. bert-tiny model zip - zip that contains the bert tiny model and BertTopic model for generation of data in section 1
4. Transformer library modifications - contains the modifications we had to make in the transformer library to incoporate CLM loss to the GAN for GPT models ONLY!

Enjoy :)
