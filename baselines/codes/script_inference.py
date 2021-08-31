#coding:utf8
#
# Run this code to get the final results reported in our ijcai paper.
from GCN import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == '__main__':
    torch.cuda.set_device(2)  # torch>=1.8
    root_path = "../../data/"
    test_index = pickle.load(open(root_path+"test_index.pickle", "rb"))
    POSITIONAL_SIZE = 36            # sequence length, 8(context)+5(candidate), 8*4+4=36
    EMBEDDING_SIZE = 128            # size of argument
    HIDDEN_SIZE = EMBEDDING_SIZE*4  # size of event
    N_LAYERS = 1
    N_HEADS = [4, 16]
    DROPOUT = float(0.0)

    model_name ='SGNN'  # 'MCer'

    test_set = Data(pickle.load(open(root_path+"vocab_index_test.data", "rb")))
    word_embedding = get_word_embedding(root_path)

    if model_name == 'MCer':
        model = ASAGCN(positional_size=POSITIONAL_SIZE,
                       vocab_size=len(word_embedding),
                       embedding_size=EMBEDDING_SIZE,
                       word_embedding=word_embedding,
                       hidden_size=HIDDEN_SIZE,
                       n_layers=N_LAYERS,
                       n_heads=N_HEADS,
                       dropout=DROPOUT)
        model = to_cuda(model)
        model.load_state_dict(torch.load('../models/MCer_best_acc_62.67.model'))
        model.eval()
    elif model_name == 'SGNN':
        model = ScaleGNN(vocab_size=len(word_embedding),
                         embedding_size=EMBEDDING_SIZE,
                         word_embedding=word_embedding,
                         hidden_size=HIDDEN_SIZE,
                         dropout=DROPOUT)
        model = to_cuda(model)
        model.load_state_dict(torch.load('../models/SGNN_best_acc_62.54.model'))
        model.eval()
    else:
        print('no models')
        exit(-1)

    test_data = test_set.all_data()
    with torch.no_grad():
        test_acc, test_result = model.predict_eval(test_data[0], test_data[1], test_data[2], test_index)
        print("Test Acc: %f" % test_acc)
