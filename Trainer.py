import torch
import torch.nn as nn
import torch.optim as op
from torch.utils.data import DataLoader

from Loadable import Loadable
from MultiplyModel import MultiplyModel


class Trainer(Loadable):

    criterion = nn.Module
    optimizer = op.Optimizer

    def __init__(self, model: MultiplyModel, padding_value: int, learning_rate: float):
        super(Trainer, self).__init__(model)
        self.padding_value = padding_value
        self.criterion = nn.MSELoss()
        self.optimizer = op.NAdam(model.parameters(), lr=learning_rate)

    def run(self, num_epochs: int, dataloader: DataLoader):
        floss = None
        for epoch in range(num_epochs):
            nb = 0
            floss = 0.0
            for batch in dataloader:

                batch = batch.to(self.model.device)
                src = batch[:, :-1]
                tgt = torch.unsqueeze(batch[:, -1],dim=1)

                self.optimizer.zero_grad()
                pad = torch.zeros(size=tgt.size(), dtype=tgt.dtype, device=self.model.device)
                pad[:,-1] = self.padding_value
                output = self.model(src, pad)

                loss = self.criterion(output, self.model.target(tgt.view(-1)))
                nb += 1
                floss += (loss.item()-floss)/nb  # mean across all batches within epoch
                loss.backward()
                self.optimizer.step()

            if floss is not None:
                if int(floss*100000000) <= 9:
                    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {floss :.12f}")
                elif int(floss*10000) <= 9:
                    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {floss :.8f}")
                else:
                    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {floss :.4f}")

        print(f"Training complete for {len(dataloader.dataset)} samples.")
        return floss
