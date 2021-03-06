import model as trf
from process import get_data
from config import *

import torch
import torch.nn as nn
import time
from torch.autograd import Variable
import torch.nn.functional as F


from torchtext import data, datasets

class Batch:
    def __init__(self, src, trg=None, pad=1):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod 
    def make_std_mask(tgt, pad=1):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            trf.subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

def train_model(epochs, model, criterion, model_opt, train_iter, save_path, print_every = 100):
    
    model.train()
    start = time.time()
    total_loss = 0
    mean_tokens = 0
    
    for epoch in range(epochs):
        for i, batch in enumerate(train_iter):
            
            src = batch.en
            tgt = batch.de
            
            # equalize sequence length of batches, originated from torchtext
            diff_ = src.size(-1) - tgt.size(-1)
            bal_pad = torch.ones(src.shape[0], abs(diff_), dtype = torch.long).cuda() # gpu setting

            if diff_ < 0:
                src = torch.cat((src, bal_pad), dim = 1)
            elif diff_ > 0:
                tgt = torch.cat((tgt, bal_pad), dim = 1)
                
            bat = Batch(src, tgt) # from train.Batch
            bat.trg_y = bat.trg_y.contiguous()
            hidden = model.forward(bat.src, bat.trg, bat.src_mask, bat.trg_mask)
            preds = model.generator(hidden)

            model_opt.zero_grad()

            loss = criterion(preds.view(-1, preds.size(-1)),
                            bat.trg_y.view(-1))
            loss.backward()

            model_opt.step()

            total_loss += loss.data
            mean_tokens += bat.ntokens / BATCH_SIZE

            del loss, preds, hidden
            torch.cuda.empty_cache()

            if i % print_every == 0:
                elapsed = time.time() - start
                print("Iteration Step: %d Loss per token: %f per Sec: %f #(tokens) : %d" %
                        (i, total_loss / mean_tokens , elapsed, mean_tokens / print_every))
                start = time.time()
                
                total_loss = 0
                mean_tokens = 0

        torch.save(model.state_dict(), save_path + str(epoch) + ".pt") # save check point
        print("Epoch Step : %d is done" %(epoch))
        