# EMDF-Net

We proposed a new script event prediction model, called *EMDF-Net*. 

## Prerequisites
- linux
- Python 3.6
- CPU or NVIDIA GPU + CUDA CuDNN
- \>= PyTorch 1.8.0 


## Paper data, models and Code

We use the same dataset as [SGNN](https://github.com/eecrazy/ConstructingNEEG_IJCAI_2018) and [MCer](https://github.com/YueAWu/MCer). You can download the dataset as follows:
- NYT dataset [[Google Drive]](https://drive.google.com/file/d/1zXTBHeBCWESX7kaAG6Q01YhUJrEl3V1j/view?usp=sharing) [[Baidu Drive]](https://pan.baidu.com/s/1pOBlOtxNIjU_ywf1_6Witg)(eg89)

The trained models can be downloaded via:
- MCer_best_acc_62.67.model [[Google Drive]](https://drive.google.com/file/d/1Pb2Yf-5BEqOPeEUxCDqtl2CXtIvuf49B/view?usp=sharing) [[Baidu Drive]](https://pan.baidu.com/s/1DVSWlC8ToA5_h-3N3NNOWA)(eapb)
- SGNN_best_acc_62.54.model [[Google Drive]](https://drive.google.com/file/d/1QKUv-2hUJ5OuSEhBAP2bWDrd100dUso8/view?usp=sharing) [[Baidu Drive]](https://pan.baidu.com/s/1e87dWfZRKPJYlSTvfPdwqA)(wf2p)
- EMDF-Net_best_acc_69.59.pth [[Google Drive]](https://drive.google.com/file/d/1F16RACy4pUtQmzV3I23nFW97xVUSVMEu/view?usp=sharing) [[Baidu Drive]](https://pan.baidu.com/s/1HnbU6i9pUNiBa8K5xQlUGg)(puaj)

**Codes Structures:**

    +--- baselines
    |   +--- codes
    |   |   +--- GCN.py            # MCer
    |   |   +--- script_inference.py   # running this file to predict
    |   |   +--- SGNN.py           # SGNN
    |   |   +--- utils.py
    |   +--- models
    |   |   +--- MCer_best_acc_62.67.model
    |   |   +--- SGNN_best_acc_62.54.model
    +--- data
    |   +--- deepwalk_128_unweighted_with_args.txt
    |   +--- dev_index.pickle
    |   +--- test_index.pickle
    |   +--- vocab.json
    |   +--- vocab_index_dev.data
    |   +--- vocab_index_test.data
    |   +--- vocab_index_train.data
    |   +--- word_embedding.npy
    +--- EMDF-Net
    |   +--- codes
    |   |   +--- config.py
    |   |   +--- script_inference.py    # running this file to predict
    |   |   +--- setup.py
    |   |   +--- src
    |   |   |   +--- datasources
    |   |   |   |   +--- ScriptData.py
    |   |   |   |   +--- __init__.py
    |   |   |   +--- models
    |   |   |   |   +--- base_model.py
    |   |   |   |   +--- FusionNet.py
    |   |   |   |   +--- my_modules.py
    |   |   |   |   +--- ScriptNet.py
    |   |   |   |   +--- __init__.py
    |   |   |   +--- util
    |   |   |   |   +--- osutils.py
    |   |   |   |   +--- tools.py
    |   +--- models
    |   |   +--- EMDF-Net_best_acc_69.59.pth
    |   +--- predicted_probability
    |   |   +--- EMDF-Net_test.scores1
    |   |   +--- EMDF-Net_test.scores2
    |   |   +--- EMDF-Net_test.scores3
    |   |   +--- EMDF-Net_test.scores4
    |   |   +--- EMDF-Net_test.scores5
    +--- other_results             ## baseline results
    |   +--- baseline_calculate.py    # running this file to predict
    |   +--- EMDF-Net_test.scores4    # EMDF-Net
    |   +--- event_comp_test.scores   # EventComp
    |   +--- MCer_test.scores2      # MCer
    |   +--- PairwiseLSTM_test.scores  # PairLSTM
    |   +--- SGNN-org.score        # original SGNN from Li,et al. https://github.com/eecrazy/ConstructingNEEG_IJCAI_2018
    |   +--- SGNN_test.scores2      # reproduced SGNN 
    |   +--- utils.py

Here we provide the PyTorch implementations of our new model EMDF-Net and two baseline models, MCer and SGNN. The SGNN and MCer are different from the original ones in the parameters and training methods, with others keeping the same. Besides, we also provide some original prediction results for several baselines, which can be directly used to produce the accuracy.



## How to run the code?

Just download the dataset, models and codes, and place them as the above settings. Then you can run the corresponding files to predict.
It is very easy! Just enjoy it!



## Questions
Please contact zhoupengpeng@bupt.edu.cn. 
