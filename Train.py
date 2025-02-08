from typing import List

import torch
import torch.nn as nn
import torch.optim as op


class Train:

    vocab_size: int
    model: nn.Module
    criterion = nn.Module
    optimizer = op.Optimizer

    def __init__(self, model: nn.Module, vocab_size: int, learning_rate: float):
        self.model = model
        self.vocab_size = vocab_size
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab_size)
        self.optimizer = op.Adam(model.parameters(), lr=learning_rate)

    def run(self, num_epochs: int, dataloader, vocab_size: int, device = None):
        floss = None
        for epoch in range(num_epochs):
            for batch in dataloader:
                batch = batch.to(device)
                src = batch[:, :-1]
                tgt = torch.unsqueeze(batch[:, -1],dim=1)

                self.optimizer.zero_grad()
                output = self.model(src, tgt)

                output = output.view(-1, vocab_size)
                tgt = tgt.view(-1)

                loss = self.criterion(output, tgt)
                floss = loss
                loss.backward()
                self.optimizer.step()

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

        print("Training complete!")
        return floss.item()

    @staticmethod
    def generate(r1: tuple, r2: tuple, op: str = " * ", eq: str = " = ") -> List[str]:
        rl = list()
        for a in range(r1[0],r1[1]+1):
            for b in range(r2[0],r2[1]+1):
                rl.append( str(a) + op + str(b) + eq + str(a*b) )
        return rl
