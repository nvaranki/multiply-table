# This is a sample Python script.
import os.path

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
    loss = trainer.run(num_epochs, dl, ds.vocab_size, device)

    # save the model
    import datetime
    dt = datetime.datetime.now().isoformat(timespec='seconds').replace("-","").replace("T","").replace(":","")
    torch.save(model.state_dict(), os.path.join("data", f"snapshot{dt}.pt"))
    # Print model's state_dict
    with open(os.path.join("data", f"snapshot{dt}.txt"), 'wt') as f:
        f.write("Model's parameters:\n")
        f.write("embed_size\t" + str(embed_size) + "\n")
        f.write("num_heads\t"  + str(num_heads)  + "\n")
        f.write("hidden_dim\t" + str(hidden_dim) + "\n")
        f.write("num_layers\t" + str(num_layers) + "\n")
        f.write("dtype\t"      + str(dtype)      + "\n")
        f.write("num_epochs\t" + str(num_epochs) + "\n")
        f.write("learning_rate\t" + str(learning_rate) + "\n")
        f.write("loss\t" + f"{loss:.4f}" + "\n")
        f.write("Model's state_dict:\n")
        nps: int = 0
        for pt in model.state_dict():
            size = model.state_dict()[pt].size()
            f.write(str(pt) + "\t" + str(size) + "\n")
            npa = 1
            for s in size:
                npa *= s
            nps += npa
        f.write(f"Total number of parameters: {nps}\n")

    print("Well done!")
