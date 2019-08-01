from process import get_data
from torchtext import data

def get_some_batch():
    "Batch 하나 가져오기"
    data_path = "dataset/"
    file_name = "train."

    txt_en, train_en = get_data(file_path = data_path + file_name + 'en',
                               field_name = 'en')

    train_loader = data.Iterator(train_en, batch_size = 3,
                                device = None, # if using GPU, type "cuda" 
                                repeat = False)

    for batch in train_loader:
        break

    a = batch.en
    return a[1]

from embedding import Embeddings, PositionalEncoding

def emb_pe(b, n_vocab = 30000):
   
    emb = Embeddings(d_model = 128, vocab = 25000)

    x = emb(b).unsqueeze(0)

    PE = PositionalEncoding(d_model = 128, dropout = .1)

    return PE(x)

    