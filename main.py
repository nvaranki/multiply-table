import torch.cuda

from MultiplyModel import MultiplyModel
from TextDataset import TextDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from Train import Train


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    ds = TextDataset(Train.generate((1,10),(1,10)))
    dl = DataLoader(ds, batch_size=5, collate_fn=lambda x: pad_sequence(x, batch_first=True, padding_value=ds.vocab_size))

    # Model parameters
    embed_size = 128
    num_heads = 2
    hidden_dim = 512
    num_layers = 2
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    model = MultiplyModel(ds.vocab_size, embed_size, num_heads, hidden_dim, num_layers, device, dtype)

    # Training loop
    num_epochs = 10
    learning_rate = 0.001
    trainer = Train(model, ds.vocab_size, learning_rate)
    weights = trainer.load()
    print("New model has been created." if weights is None
          else f"Loaded last saved weights from \"{weights}\" into the model.")
    loss = trainer.run(num_epochs, dl, ds.vocab_size, device)
    weights = trainer.save(embed_size=embed_size, num_heads=num_heads, hidden_dim=hidden_dim, num_layers=num_layers,
                           dtype=dtype, num_epochs=num_epochs, learning_rate=learning_rate, loss=loss)
    print(f"The model weights saved to \"{weights}\".")

    print("Well done!")
