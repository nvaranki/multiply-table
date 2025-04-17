import json
from typing import List, Union

import os.path
import torch
import torch.nn as nn
import torch.optim as op
from torch.utils.data import Dataset
from json import JSONEncoder
from BinaryDecoder import BinaryDecoder, DecoderLoss


class EncodeTensor(JSONEncoder, Dataset):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        else:
            return super(EncodeTensor, self).default(obj)


class Train:

    vocab_size: int
    model: nn.Module
    criterion = nn.Module
    optimizer = op.Optimizer

    def __init__(self, model: nn.Module, vocab_size: int, padding_value: int, learning_rate: float):
        self.model = model
        self.vocab_size = vocab_size
        self.padding_value = padding_value
        self.criterion = nn.MSELoss()
        self.optimizer = op.NAdam(model.parameters(), lr=learning_rate)  # Epoch [400/400], Loss: 0.6088 snapshot20250414182153

    def run(self, num_epochs: int, dataloader, vocab_size: int, device = None):
        floss = None
        for epoch in range(num_epochs):
            nb = 0
            floss = 0.0
            for batch in dataloader:
                batch = batch.to(device)
                src = batch[:, :-1]
                tgt = torch.unsqueeze(batch[:, -1],dim=1)

                self.optimizer.zero_grad()
                # with torch.no_grad():
                pad = torch.zeros(size=tgt.size(), dtype=tgt.dtype)
                pad[:,-1] = self.padding_value
                output = self.model(src, pad.to(device=device))

                output = output.view(-1, self.model.embed_size)
                tgt = tgt.view(-1)

                loss = self.criterion(output, DecoderLoss.vector(tgt,self.model.embed_size))
                nb += 1
                floss += (loss.item()-floss)/nb  # mean across all batches
                loss.backward()
                self.optimizer.step()

            if floss is not None:
                loss_item = floss
                if int(loss_item*100000000) == 0:
                    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_item :.12f}")
                elif int(loss_item*10000) == 0:
                    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_item :.8f}")
                else:
                    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_item :.4f}")

        print("Training complete!")
        return floss

    @staticmethod
    def generate(r1: tuple, r2: tuple, op: List[str] = " * ", eq: List[str] = " = ") -> List[str]:
        rl = list()
        # TODO "six times two equals twelve"
        for o in op:
            for e in eq:
                for a in range(r1[0],r1[1]+1):
                    for b in range(r2[0],r2[1]+1):
                        rl.append( str(a) + o + str(b) + e + str(a*b) )
        return rl

    def load(self, dir="data") -> Union[str,None]:
        """load the last saved model"""
        ss = [n for n in os.listdir(dir) if n.startswith("snapshot") and n.endswith(".pt")]
        if len(ss) > 0:
            ss.sort(reverse=True)
            fn = os.path.join(dir, ss[0])
            self.model.load_state_dict(torch.load(fn, weights_only=True))
            self.model.eval()
            return fn
        else:
            return None

    def save(self, dir="data", **kwa) -> str:
        """save the model with notes"""

        import datetime
        dt = datetime.datetime.now().isoformat(timespec='seconds').replace("-","").replace("T","").replace(":","")
        mfn = os.path.join(dir, f"snapshot{dt}.pt")     # model
        tfn = os.path.join(dir, f"snapshot{dt}.txt")    # memo
        bwp = os.path.join(dir, f"snapshot{dt}w.json")  # weighs
        bwg = os.path.join(dir, f"snapshot{dt}g.json")  # gradients
        # btp = os.path.join(dir, f"snapshot{dt}t.json")  # tokens
        bvp = os.path.join(dir, f"snapshot{dt}v.json")  # vocabulary

        # save numeric data
        torch.save(self.model.state_dict(), mfn)
        with open(bwp, "wt") as f:
            json.dump(self.model.state_dict(), f, cls=EncodeTensor)
        grads = {}
        for k, v in self.model.state_dict(keep_vars=True).items():
            grads[k+".grad"] = v.grad
        with open(bwg, "wt") as f:
            json.dump(grads, f, cls=EncodeTensor)
        # with open(btp, "wt") as f:
        #     json.dump(kwa["tokens"], f, cls=EncodeTensor)
        with open(bvp, "wt") as f:
            json.dump(kwa["vocab"], f, cls=EncodeTensor)

        # Print model's state_dict
        with open(tfn, 'wt') as f:
            f.write("Model's parameters:\n")
            f.write("embed_size\t" + str(kwa["embed_size"]) + "\n")
            f.write("num_heads\t"  + str(kwa["num_heads"])  + "\n")
            f.write("hidden_dim\t" + str(kwa["hidden_dim"]) + "\n")
            f.write("num_layers\t" + str(kwa["num_layers"]) + "\n")
            f.write("dtype\t"      + str(kwa["dtype"])      + "\n")
            f.write("num_epochs\t" + str(kwa["num_epochs"]) + "\n")
            f.write("learning_rate\t" + str(kwa["learning_rate"]) + "\n")
            f.write("loss\t" + f"{kwa["loss"]:.15f}" + "\n")
            f.write("Model's state_dict:\n")
            nps: int = 0
            for pt in self.model.state_dict():
                size = self.model.state_dict()[pt].size()
                f.write(str(pt) + "\t" + str(size) + "\n")
                npa = 1
                for s in size:
                    npa *= s
                nps += npa
            f.write(f"Total number of parameters: {nps}\n")

        return mfn
