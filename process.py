from torchtext import data, datasets

'''
MIN_FREQ : minimum co-occurence for vocab. dictionary
DEVICE_SET : if using GPU, type "cuda"
BATCH_SIZE : batch size
'''

def get_data(file_path, field_name, MIN_FREQ = 2, DEVICE_SET = None):
    
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"
                
    data_field = data.Field(sequential = True, use_vocab = True,
                        batch_first = True,tokenize=str.split, 
                        init_token = BOS_WORD,                   
                        eos_token = EOS_WORD, pad_token=BLANK_WORD)

    tabular_set = data.TabularDataset(path= file_path, 
                                 fields = [(field_name, data_field)],
                                 format = 'tsv')

    data_field.build_vocab(tabular_set, min_freq = MIN_FREQ)

    return data_field, tabular_set