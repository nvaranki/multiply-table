import torch
from torch.utils.data import DataLoader

from Loadable import Loadable
from MultiplyModel import MultiplyModel


class Runner(Loadable):

    def __init__(self, model: MultiplyModel, padding_value: int):
        super(Runner, self).__init__(model)
        self.padding_value = padding_value

    def run(self, dataloader: DataLoader):
        with torch.no_grad():
            for batch in dataloader:

                batch = batch.to(self.model.device)
                src = batch[:, :-1]
                tgt = torch.unsqueeze(batch[:, -1], dim=1)

                pad = torch.zeros(size=tgt.size(), dtype=tgt.dtype, device=self.model.device)
                pad[:,-1] = self.padding_value
                output = self.model(src, pad)

                output = output.view(-1, self.model.vocab_size)
                tgt = tgt.view(-1)

                tokens = output.max(dim=-1).indices
                fail = tokens - tgt
                if fail.abs().to(dtype=torch.bool).any().item():
                    print(f"Error: found={tokens.detach().cpu().numpy()} expected={tgt.detach().cpu().numpy()} \n\tsource={src.detach().cpu().numpy()}")
        print(f"{len(dataloader.dataset)} records were processed.")
