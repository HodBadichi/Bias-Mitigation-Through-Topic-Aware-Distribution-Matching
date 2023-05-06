import pandas as pd 
import wandb


api = wandb.Api()
entity, project = "best-bert", "Discriminator_test"  
runs = api.runs(entity + "/" + project) 
for run in runs: 
    print(run.name)
    if run.state != "finished":
       continue
    history = run.scan_history()
    df = pd.DataFrame(history)
    important_columns = df.filter(regex='train|test|val').columns
    stats = df[important_columns].describe().to_dict()
    # print(df["discriminator/val_dataset_accuracy"])
    for key in stats:
        for stat in stats[key]:
            if stat!="count":
                run.summary[key + "_" + stat] = stats[key][stat]
    run.update()

