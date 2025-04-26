import json
import os.path
from json import JSONEncoder
from typing import Union

import torch

from MultiplyModel import MultiplyModel


class EncodeTensor(JSONEncoder, torch.utils.data.Dataset):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        else:
            return super(EncodeTensor, self).default(obj)


class Loadable:

    model: MultiplyModel

    def __init__(self, model: MultiplyModel):
        self.model = model

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
