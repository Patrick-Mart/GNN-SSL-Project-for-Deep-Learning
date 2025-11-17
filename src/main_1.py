
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from pytorch_lightning.loggers import CSVLogger

from data import MAG240MInductiveDataset
from model_1 import get_model
from ogb.nodeproppred import Evaluator

class SimpleEvaluator:
    def __init__(self, num_classes=None):
        pass
    
    def eval(self, input_dict):
        y_true = input_dict['y_true'].cpu()
        y_pred = input_dict['y_pred'].cpu().argmax(dim=-1)
        
        acc = (y_pred == y_true).float().mean().item()
        return {'acc': acc}




class MAG240MLightning(pl.LightningModule):
    def __init__(self, model, lr=0.001):
        super().__init__()
        self.model = model
        # self.evaluator = Evaluator(name='ogbn-mag240m')
        self.evaluator = SimpleEvaluator()
        self.lr = lr

    def forward(self, x_dict, edge_index_dict):
        return self.model(x_dict, edge_index_dict)['paper']

    def training_step(self, batch, batch_idx):
        y_hat = self(batch.x_dict, batch.edge_index_dict)
        y = batch['paper'].y
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, batch_size=y_hat.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        # For true inductive evaluation, you'd load val/test separately
        # Here we just monitor training stability
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return [optimizer], [scheduler]


def main():
    # Load inductive data
    dataset = MAG240MInductiveDataset()
    data = dataset.get_data()

    # Model
    model = get_model(data, hidden_dim=1024)

    # Data loader (only trains on train papers)
    train_loader = NeighborLoader(
        data,
        num_neighbors=[25, 20, 15],
        batch_size=1024,
        input_nodes=('paper', None),  # all remaining papers are train
        shuffle=True,
        num_workers=10,
        persistent_workers=True,
    )

    # Lightning module
    lit_model = MAG240MLightning(model, lr=0.001)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator='gpu',
        devices='auto',
        strategy='ddp' if torch.cuda.device_count() > 1 else None,
        precision='16-mixed',
        logger=CSVLogger("logs/", name="mag240m_inductive"),
        log_every_n_steps=10,
        enable_checkpointing=True,
    )

    print(f"Training on {data['paper'].num_nodes:,} papers (inductive setting)")
    trainer.fit(lit_model, train_loader)

    print("Training complete! Model ready for inductive inference on val/test sets.")


main()