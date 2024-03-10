import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch

for size in ['base', 'large']:
    model = AutoModel.from_pretrained('roberta-'+size)
    tokenizer = AutoTokenizer.from_pretrained("roberta-"+size)

    fnames = ["2D_CFD_Rand_M0.1_Eta0.1_Zeta0.1_periodic_128_Train", "1D_CFD_Rand_Eta0.1_Zeta0.1_periodic_Train", "1D_Burgers_Sols_Nu0.001",  "1D_Advection_Sols_beta0.4", "1D_diff-sorp_NA_NA", "ReacDiff_Nu0.5_Rho1.0", '2D_rdb_NA_NA', "2D_diff-react_NA_NA"]
    sents = ["2D Navier-Stokes M=0.1 Eta=0.1 Zeta=0.1 Periodic","1D Navier-Stokes Eta=0.1 Zeta=0.1 Periodic", "1D Burgers' Equation Nu=0.001", "1D Advection Beta=0.4", "1D Diffusion-Sorption", "1D Diffusion-Reaction Nu=0.5 Rho=1.0", '2D Shallow-Water', "2D Diffusion-Reaction"]

    for i, sent in enumerate(sents):
        tok = tokenizer(sent)["input_ids"]
        inid = model.embeddings(torch.from_numpy(np.array(tok)).unsqueeze(0)).detach().cpu().numpy()
        np.save("datasets/" + fnames[i] +"_" + size + "_text.npy",inid)