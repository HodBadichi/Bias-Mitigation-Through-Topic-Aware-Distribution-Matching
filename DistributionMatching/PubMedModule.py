import pytorch_lightning as pl
class GAN(pl.LightningModule):
    def __init__(self):
        pass
    def forward(self):
        pass
    def step(self,batch,optimizer_idx,name):
        # if optimzer_index == 0 :
        # batch is dict of {more_women:[d1...dn], less_women:[d1...dn]}
        # return dict of loss and everything we want to log
        pass
    def training_step(self):
        pass
    def validation_step(self):
        pass
    def test_step(self):
        pass

    def configure_optimizers(self):
        pass