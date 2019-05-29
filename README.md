# rnn-bias
Code for "Does syntax need to grow on trees? Sources of inductive bias in sequence to sequence networks"

# Dependencies
I ran this with PyTorch version 0.4.0, but other versions may well work. It should run on either a GPU or CPU (though at least some experiments will be markedly faster on a GPU).

# Data
Files ending in `.train`, `.dev`, `.test`, and `.gen` are training, development, test, and generalization sets, respectively. Files starting `agr` are the data for the question formation task, and file starting with `tense` are the data for the tense reinflection task.

# Running the code
Each experiment is broken into 2 parts (training and testing) as follows:

```
python seq2seq.py task1 task2 model attention lr hs seed
python test.py task1 task2 model attention lr hs > outfile
```

Here are descriptions of the arguments for seq2seq.py, the script that trains models:
- `task1`: The training task. Can either be `agr` (for question formation) or `tense` (for tense reinflection)
- `task2`: This is a duplicate of the first `task` option and should be identical to whatever you choose for `task`. (To-do: Get rid of this, since it's redundant).
- `model`: The type of model to be used. Options (a given option will select that option as the architecture for both the encoder and the decoder, unless otherwise noted):
  * `SRN`: Simple Recurrent Network 
  * `GRU`: Gated Recurrent Unit
  * `LSTM`: Long Short-Term Memory unit
  * `MyLSTM`: (Ignore this - not used in the paper)
  * `LSTMSqueeze`: (Ignore this - not used in the paper)
  * `LSTMBob`: Squashed LSTM
  * `ONLSTM`: Ordered Neurons LSTM
  * `GRUUnsqueeze`: (Ignore this - not used in the paper)
  * `GRUUnsqueeze2`: (Ignore this - not used in the paper)
  * `GRUBob`: Unsquashed GRU
  * `TREE`: (Ignore this - not used in the paper)
  * `TREEENC`: (Ignore this - not used in the paper)
  * `TREEDEC`: (Ignore this - not used in the paper)
  * `TREEBOTH`: (Ignore this - not used in the paper)
  * `TREENew`: (Ignore this - not used in the paper)
  * `TREEENCNew`: (Ignore this - not used in the paper)
  * `TREENOPRE`: (Ignore this - not used in the paper)
  * `TREEENCNOPRE`: (Ignore this - not used in the paper)
  * `TREEDECNOPRE`: Model with a linear GRU encoder and a tree-GRU decoder
  * `TREEBOTHNOPRE`: (Ignore this - not used in the paper)
  * `TREENewNOPRE`: Model with a tree-GRU encoder and a tree-GRU decoder
  * `TREEENCNewNOPRE`: Model with a tree-GRU encoder and a linear GRU decoder
  * `ONLSTMPROC`: (Ignore this - not used in the paper)
- `attention`: The type of attention used by the model. Options:
  * `0`: No attention
  * `1`: Location-based attention
  * `2`: Content-based attention
- `lr`: The learning rate
- `hs`: The size of the hidden state
- `seed`: The random seed

To test a trained model (or models), use `test.py`. This script will compile together the results from all random initializations of the architecture that you specify with the arguments; for example, even if you ran 100 different random initializations of a model, you would still only need to run `test.py` once because it would gather the results for all 100 initializations. All of the arguments for `test.py` are the same as for `seq2seq.py`. Note: `test.py` is only for the question formation task. To test a model trained on tense reinflection, use `test_tense.py` instead. The output will print to the command line, unless you specify an output file (as illustrated above with `> outfile').

Example: Suppose you wanted to test a GRU without attention on the question formation task with 5 random initializations, a hidden size of 256, and a learning rate of 0.001. The commands you would run to do this would be as follows (note that the first 5 lines are all the same except for the random seeds, to give you 5 different initializations of the same model):

```
python seq2seq.py agr agr GRU 0 0.001 256 0
python seq2seq.py agr agr GRU 0 0.001 256 1
python seq2seq.py agr agr GRU 0 0.001 256 2
python seq2seq.py agr agr GRU 0 0.001 256 3
python seq2seq.py agr agr GRU 0 0.001 256 4

python test.py agr agr GRU 0 0.001 256 > outfile
```

# Understanding the test output
The output of the test script is currently not very user-friendly (I'll be cleaning it up soon). The two most important things indicated in the output are the two metrics we focus on in the paper, namely test set full-sentence accuracy and generalization set first-word accuracy. Here is how to find each of these in the test script output:
- Test set full-sentence accuracy: If you search for "Overall test correct:", that will be the start of a block of results about test set full-sentence accuracy. At the end of that block will be 4 lines starting with "Mean:", "Median:", "Mean10:", and "Median10:". The first 2 of those are the most relevant: "Mean:" gives the mean test set full-sentence accuracy across all the random initializations, and "Median:" gives the median.
- Generalization set first-word accuracy: Same as above, except that the relevant block for this one is the one starting with "Overall gen first word correct aux:"
  



