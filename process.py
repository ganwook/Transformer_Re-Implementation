from torchtext import data, datasets

'''
MIN_FREQ : minimum co-occurence for vocab. dictionary
DEVICE_SET : if using GPU, type "cuda"
BATCH_SIZE : batch size
'''

def get_data(file_path, MIN_FREQ = 2, DEVICE_SET = None):
    
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    PAD_WORD = "<pad>"
                
    field_en = data.Field(sequential = True, use_vocab = True,
                    batch_first = True,tokenize=str.split, 
                    init_token = BOS_WORD,                   
                 eos_token = EOS_WORD, pad_token=PAD_WORD)
    
    field_de = data.Field(sequential = True, use_vocab = True,
                    batch_first = True, tokenize=str.split, 
                    init_token = BOS_WORD,                   
                    eos_token = EOS_WORD, pad_token=PAD_WORD)

    trn = datasets.TranslationDataset(path = file_path, exts = ('en', 'de'),
                           fields = [('en', field_en), ('de', field_de)])

    field_en.build_vocab(trn.en, min_freq = MIN_FREQ)
    field_de.build_vocab(trn.de, min_freq = MIN_FREQ)

    return field_en, field_de, trn