from DistributionMatching.PubMed.PubMedModule import PubMedModule

if __name__ == '__main__':
    dl = PubMedModule()
    print("Done init")
    dl.prepare_data()
    print("Done prepare_Data")
    dl.setup()
    print("Done Setup")
    for batch in dl.train_dataloader():
        print(batch)
