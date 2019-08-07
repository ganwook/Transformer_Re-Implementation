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

def train_model(epochs, model, criterion, model_opt, train_iter, print_every = 100):
    model.train()
    start = time.time()
    total_loss = 0
	
    for epoch in range(epochs):
        for i, batch in enumerate(train_iter):
            
            src = batch.en
            tgt = batch.de
            
            # equalize sequence length of batches, originated from torchtext
            diff_ = src.size(-1) - tgt.size(-1)
            bal_pad = torch.ones(BATCH_SIZE, abs(diff_), dtype = torch.long)

            if diff_ < 0:
                src = torch.cat((src, bal_pad), dim = 1)
            elif diff_ > 0:
                tgt = torch.cat((tgt, bal_pad), dim = 1)
                
            bat = Batch(src, tgt) # from train.Batch
            
            hidden = model.forward(bat.src, bat.trg, bat.src_mask, bat.trg_mask)
            preds = model.generator(hidden)

            model_opt.zero_grad()

            loss = criterion(preds.contiguous().view(-1, preds.size(-1)),
                            bat.trg_y.contiguous().view(-1))
            loss.backward()

            model_opt.step()

            total_loss += loss.data

            if i % print_every == 0:
                elapsed = time.time() - start
                print("Iteration Step: %d Loss: %f per Sec: %f" %
                        (i, total_loss / print_every, elapsed))
                start = time.time()
				total_loss = 0
				
		print("Epoch Step : %d is done" %(epcoh))