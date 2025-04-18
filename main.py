import argparse

import torch.cuda
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from MultiplyModel import MultiplyModel
from Runner import Runner
from TextDataset import TextDataset
from Trainer import Trainer

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # application parameters
    cli = argparse.ArgumentParser(description="Multiplying Table")
    cli.add_argument("--new", action='store_const', const=True, required=False, default=False,
                     metavar="NEW", help="start training with blank model")
    cli.add_argument("--continue", action='store_const', const=True, required=False, default=False,
                     metavar="CONTINUE", help="continue training last model")
    args = cli.parse_args()

    # Train/test data TODO load blank only, use 0 as padding index
    ds = TextDataset(TextDataset.generate((1, 10), (1, 10), [" * ", " times "], [" = ", " equals "]))
    dl = DataLoader(ds, batch_size=5, collate_fn=lambda x: pad_sequence(x, batch_first=True, padding_value=ds.vocab_size))

    # Model parameters
    embed_size = 16  # 20 good more  # 24 even better
    num_heads = 1  # Loss: 0.0462 TODO 4 Loss: 0.1402
    hidden_dim = 16
    num_layers = 1  # last three were useless 4  # faster than 8 Loss: 0.0476
    device = torch.device("cuda:0")
    dtype = torch.float
    model = MultiplyModel(ds.vocab_size, embed_size, num_heads, hidden_dim, num_layers, device, dtype)

    if args.new or args.__dict__["continue"]:

        # Training loop
        print("Running the model in training mode.")
        num_epochs = 400
        learning_rate = 0.0005
        trainer = Trainer(model, ds.vocab_size, learning_rate)
        backup = trainer.load("data") if not args.new else None
        print("New model has been created." if backup is None
              else f"Last saved weights are loaded from the \"{backup}\" into the model.")
        loss = trainer.run(num_epochs, dl)
        backup = trainer.save("data", embed_size=embed_size, num_heads=num_heads, hidden_dim=hidden_dim, num_layers=num_layers,
                              dtype=dtype, num_epochs=num_epochs, learning_rate=learning_rate, loss=loss, vocab=ds.vocab)
        print(f"The model weights are saved to \"{backup}\".")

    else:

        # Running loop
        print("Running the model in inference mode.")
        runner = Runner(model, ds.vocab_size)
        weights = runner.load("data")
        if weights is None:
            print(f"Error: No model found at \"{"data"}\".")
        else:
            print(f"Weights are loaded from the \"{weights}\" into the model.")
            runner.run(dl)
        pass

    print("Well done!")
