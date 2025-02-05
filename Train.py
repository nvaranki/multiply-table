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

    def run(self, num_epochs: int, dataloader, vocab_size: int):
        for epoch in range(num_epochs):
            for batch in dataloader:
                src = batch[:, :-1]
                tgt = batch[:, 1:]

                self.optimizer.zero_grad()
                output = self.model(src, tgt)

                output = output.view(-1, vocab_size)
                tgt = tgt.view(-1)

                loss = self.criterion(output, tgt)
                loss.backward()
                self.optimizer.step()

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

        print("Training complete!")
