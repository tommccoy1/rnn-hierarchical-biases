# rnn-bias
Code for "Does syntax need to grow on trees? Sources of inductive bias in sequence to sequence networks"

# Dependencies
We ran this with PyTorch version 0.4.0, but other versions may well work. It should run on either a GPU or CPU (though at least some experiments will be markedly faster on a GPU).

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
  * `LSTMBob`: Squashed LSTM
  * `ONLSTM`: Ordered Neurons LSTM
  * `GRUBob`: Unsquashed GRU
  * `TREEDECNOPRE`: Model with a linear GRU encoder and a tree-GRU decoder
  * `TREENewNOPRE`: Model with a tree-GRU encoder and a tree-GRU decoder
  * `TREEENCNewNOPRE`: Model with a tree-GRU encoder and a linear GRU decoder
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

# Notes on running

The non-tree-based models run reasonably quickly; on an NVIDIA k80 GPU, these models will converge in roughly 30 minutes to 1 hour (on a CPU, they train more slowly, but should still converge within a few hours). Models with content-based attention are significantly slower than models with no attention or position-based attention (but should still converge within a few hours).

The tree-based models take much longer to train (over 1 day, whether on a CPU or GPU; GPUs do not bring about much speedup for these models because their batches are not implemented in a way that GPUs can take advantage of).

# Example

This repo contains the output of one small example. The example was created by running the commands in `GRU_agr_1_0.01_256.scr`. The first 2 commands in that script train 2 instances of a GRU with location-based attention, a hidden size of 256, and a learning rate of 0.01 on the task of question formation. These commands create the subdirectories `agr_GRU_1_0.01_256_0` and `agr_GRU_1_0.01_256_1`. Each of these subdirectories contains the saved weights of the model being trained by that command (these weights are split into 2 files, one for the encoder's weights and one for the decoder's weights). The remaining files in these subdirectories are empty and only serve as indicators of progress during training.

The third line in `GRU_agr_1_0.01_256.scr` then tests the trained models and outputs the results to `test_agr_GRU_1_0.01_256.out`. There is a lot of information in `test_agr_GRU_1_0.01_256.out`, but the most important data are that the test set full-sentence accuracy had a median of 0.974 and that the generalization set first-word accuracy had a median of 0.926.

# Basic description of the code

# How to replicate the experiments in the paper

### Variants on sequential RNNs (Section 3.3 / Figure 4)

Run each of the following lines 100 times (the code will automatically generate a separate folder for each run):
- `python seq2seq.py --encoder SRN --decoder SRN --task question --attention none --lr 0.001 --hs 256`
- `python seq2seq.py --encoder SRN --decoder SRN --task question --attention location --lr 0.001 --hs 256`
- `python seq2seq.py --encoder SRN --decoder SRN --task question --attention content --lr 0.001 --hs 256`
- `python seq2seq.py --encoder GRU --decoder GRU --task question --attention none --lr 0.001 --hs 256`
- `python seq2seq.py --encoder GRU --decoder GRU --task question --attention location --lr 0.001 --hs 256`
- `python seq2seq.py --encoder GRU --decoder GRU --task question --attention content --lr 0.001 --hs 256`
- `python seq2seq.py --encoder LSTM --decoder LSTM --task question --attention none --lr 0.001 --hs 256`
- `python seq2seq.py --encoder LSTM --decoder LSTM --task question --attention location --lr 0.001 --hs 256`
- `python seq2seq.py --encoder LSTM --decoder LSTM --task question --attention content --lr 0.001 --hs 256`
  
Then run these evaluation steps (just once):
- `python test_question.py --encoder SRN --decoder SRN --task question --attention none --lr 0.001 --hs 256 > SRN_SRN_question_none_0.001_256.results`
- `python test_question.py --encoder SRN --decoder SRN --task question --attention location --lr 0.001 --hs 256 > SRN_SRN_question_location_0.001_256.results`
- `python test_question.py --encoder SRN --decoder SRN --task question --attention content --lr 0.001 --hs 256 > SRN_SRN_question_content_0.001_256.results`
- `python test_question.py --encoder GRU --decoder GRU --task question --attention none --lr 0.001 --hs 256 > GRU_GRU_question_none_0.001_256.results`
- `python test_question.py --encoder GRU --decoder GRU --task question --attention location --lr 0.001 --hs 256 > GRU_GRU_question_location_0.001_256.results`
- `python test_question.py --encoder GRU --decoder GRU --task question --attention content --lr 0.001 --hs 256 > GRU_GRU_question_content_0.001_256.results`
- `python test_question.py --encoder LSTM --decoder LSTM --task question --attention none --lr 0.001 --hs 256 > LSTM_LSTM_question_none_0.001_256.results`
- `python test_question.py --encoder LSTM --decoder LSTM --task question --attention location --lr 0.001 --hs 256 > LSTM_LSTM_question_location_0.001_256.results`
- `python test_question.py --encoder LSTM --decoder LSTM --task question --attention content --lr 0.001 --hs 256 > LSTM_LSTM_question_content_0.001_256.results`


### Squashing experiments (Section 3.4 / Figure 5)

Run each of these training scripts 100 times (the plain GRU is the squashed GRU; the plain LSTM is the unsquashed LSTM):
- `python seq2seq.py --encoder GRU --decoder GRU --task question --attention location --lr 0.001 --hs 256`
- `python seq2seq.py --encoder UnsquashedGRU --decoder UnsquashedGRU --task question --attention location --lr 0.001 --hs 256`
- `python seq2seq.py --encoder LSTM --decoder LSTM --task question --attention location --lr 0.001 --hs 256`
- `python seq2seq.py --encoder SquashedLSTM --decoder SquashedLSTM --task question --attention location --lr 0.001 --hs 256`

Then run each of these evaluation lines (just once each):
- `python test_question.py --encoder GRU --decoder GRU --task question --attention location --lr 0.001 --hs 256 > GRU_GRU_question_location_0.001_256.results`
- `python test_question.py --encoder UnsquashedGRU --decoder UnsquashedGRU --task question --attention location --lr 0.001 --hs 256 > UnsquashedGRU_UnsquashedGRU_question_location_0.001_256.results`
- `python test_question.py --encoder LSTM --decoder LSTM --task question --attention location --lr 0.001 --hs 256 > LSTM_LSTM_question_location_0.001_256.results`
- `python test_question.py --encoder SquashedLSTM --decoder SquashedLSTM --task question --attention location --lr 0.001 --hs 256 > SquashedLSTM_SquashedLSTM_question_location_0.001_256.results`

### Ordered Neurons model for question formation (Section 4.1)

Run this training step 100 times:
- `python seq2seq.py --encoder ONLSTM --decoder ONLSTM --task question --attention none --lr 0.001 --hs 256`

Run this evaluation step once:
- `python test_question.py --encoder ONLSTM --decoder ONLSTM --task question --attention none --lr 0.001 --hs 256 > ONLSTM_ONLSTM_question_none_0.001_256.results`

### Tree-GRUs for question formation (Section 4.2 / Figure 7)
Run the following training steps 100 times each:
- `python seq2seq.py --encoder GRU --decoder GRU --task question --attention none --lr 0.001 --hs 256`
- `python seq2seq.py --encoder GRU --decoder Tree --task question --attention none --lr 0.001 --hs 256`
- `python seq2seq.py --encoder Tree --decoder GRU --task question --attention none --lr 0.001 --hs 256`
- `python seq2seq.py --encoder Tree --decoder Tree --task question --attention none --lr 0.001 --hs 256`

Run the following evaluations once each:
- `python test_question.py --encoder GRU --decoder GRU --task question --attention none --lr 0.001 --hs 256 > GRU_GRU_question_none_0.001_256.results`
- `python test_question.py --encoder GRU --decoder Tree --task question --attention none --lr 0.001 --hs 256 > GRU_Tree_question_none_0.001_256.results`
- `python test_question.py --encoder Tree --decoder GRU --task question --attention none --lr 0.001 --hs 256 > Tree_GRU_question_none_0.001_256.results`
- `python test_question.py --encoder Tree --decoder Tree --task question --attention none --lr 0.001 --hs 256 > Tree_Tree_question_none_0.001_256.results`


### Tense reinflection (Section 5.1 / Figure 8)
Run the following training steps 100 times each:
- `python seq2seq.py --encoder SRN --decoder SRN --task tense --attention none --lr 0.001 --hs 256`
- `python seq2seq.py --encoder SRN --decoder SRN --task tense --attention location --lr 0.001 --hs 256`
- `python seq2seq.py --encoder SRN --decoder SRN --task tense --attention content --lr 0.001 --hs 256`
- `python seq2seq.py --encoder GRU --decoder GRU --task tense --attention none --lr 0.001 --hs 256`
- `python seq2seq.py --encoder GRU --decoder GRU --task tense --attention location --lr 0.001 --hs 256`
- `python seq2seq.py --encoder GRU --decoder GRU --task tense --attention content --lr 0.001 --hs 256`
- `python seq2seq.py --encoder LSTM --decoder LSTM --task tense --attention none --lr 0.001 --hs 256`
- `python seq2seq.py --encoder LSTM --decoder LSTM --task tense --attention location --lr 0.001 --hs 256`
- `python seq2seq.py --encoder LSTM --decoder LSTM --task tense --attention content --lr 0.001 --hs 256`
- `python seq2seq.py --encoder ONLSTM --decoder ONLSTM --task tense --attention none --lr 0.001 --hs 256`
- `python seq2seq.py --encoder Tree --decoder Tree --task tense --attention none --lr 0.001 --hs 256`

Run the following evaluations once each:
- `python test_tense.py --encoder SRN --decoder SRN --task tense --attention none --lr 0.001 --hs 256 > SRN_SRN_tense_none_0.001_256.results`
- `python test_tense.py --encoder SRN --decoder SRN --task tense --attention location --lr 0.001 --hs 256 > SRN_SRN_tense_location_0.001_256.results`
- `python test_tense.py --encoder SRN --decoder SRN --task tense --attention content --lr 0.001 --hs 256 > SRN_SRN_tense_content_0.001_256.results`
- `python test_tense.py --encoder GRU --decoder GRU --task tense --attention none --lr 0.001 --hs 256 > GRU_GRU_tense_none_0.001_256.results`
- `python test_tense.py --encoder GRU --decoder GRU --task tense --attention location --lr 0.001 --hs 256 > GRU_GRU_tense_location_0.001_256.results`
- `python test_tense.py --encoder GRU --decoder GRU --task tense --attention content --lr 0.001 --hs 256 > GRU_GRU_tense_content_0.001_256.results`
- `python test_tense.py --encoder LSTM --decoder LSTM --task tense --attention none --lr 0.001 --hs 256 > LSTM_LSTM_tense_none_0.001_256.results`
- `python test_tense.py --encoder LSTM --decoder LSTM --task tense --attention location --lr 0.001 --hs 256 > LSTM_LSTM_tense_location_0.001_256.results`
- `python test_tense.py --encoder LSTM --decoder LSTM --task tense --attention content --lr 0.001 --hs 256 > LSTM_LSTM_tense_content_0.001_256.results`
- `python test_tense.py --encoder ONLSTM --decoder ONLSTM --task tense --attention none --lr 0.001 --hs 256 > ONLSTM_ONLSTM_tense_none_0.001_256.results`
- `python test_tense.py --encoder Tree --decoder Tree --task tense --attention none --lr 0.001 --hs 256 > Tree_Tree_tense_none_0.001_256.results`


### Unambiguous training sets (Section 6)
Run the following training steps 100 times each:
- `python seq2seq.py --encoder GRU --decoder GRU --task question_main --attention none --lr 0.001 --hs 256`
- `python seq2seq.py --encoder GRU --decoder GRU --task question_first --attention none --lr 0.001 --hs 256`
- `python seq2seq.py --encoder Tree --decoder Tree --task question_main --attention none --lr 0.001 --hs 256`
- `python seq2seq.py --encoder Tree --decoder Tree --task question_first --attention none --lr 0.001 --hs 256`
- `python seq2seq.py --encoder GRU --decoder GRU --task tense_recent --attention none --lr 0.001 --hs 256`
- `python seq2seq.py --encoder GRU --decoder GRU --task tense_subject --attention none --lr 0.001 --hs 256`
- `python seq2seq.py --encoder Tree --decoder Tree --task tense_recent --attention none --lr 0.001 --hs 256`
- `python seq2seq.py --encoder Tree --decoder Tree --task tense_subject --attention none --lr 0.001 --hs 256`

Run the following evaluations once each:
- `python test_question.py --encoder GRU --decoder GRU --task question_main --attention none --lr 0.001 --hs 256 > GRU_GRU_question_main_none_0.001_256.results`
- `python test_question.py --encoder GRU --decoder GRU --task question_first --attention none --lr 0.001 --hs 256 > GRU_GRU_question_first_none_0.001_256.results`
- `python test_question.py --encoder Tree --decoder Tree --task question_main --attention none --lr 0.001 --hs 256 > Tree_Tree_question_main_none_0.001_256.results`
- `python test_question.py --encoder Tree --decoder Tree --task question_first --attention none --lr 0.001 --hs 256 > Tree_Tree_question_first_none_0.001_256.results`
- `python test_tense.py --encoder GRU --decoder GRU --task tense_recent --attention none --lr 0.001 --hs 256 > GRU_GRU_tense_recent_none_0.001_256.results`
- `python test_tense.py --encoder GRU --decoder GRU --task tense_subject --attention none --lr 0.001 --hs 256 > GRU_GRU_tense_subject_none_0.001_256.results`
- `python test_tense.py --encoder Tree --decoder Tree --task tense_recent --attention none --lr 0.001 --hs 256 > Tree_Tree_tense_recent_none_0.001_256.results`
- `python test_tense.py --encoder Tree --decoder Tree --task tense_subject --attention none --lr 0.001 --hs 256 > Tree_Tree_tense_subject_none_0.001_256.results`


### Tree structure vs. tree information (Section 7 / Figure 9)
Run the following training steps 100 times each:
- `python seq2seq.py --encoder GRU --decoder GRU --task question --attention none --lr 0.001 --hs 256`
- `python seq2seq.py --encoder GRU --decoder GRU --task question_bracket --attention none --lr 0.001 --hs 256 --patience 6`
- `python seq2seq.py --encoder Tree --decoder Tree --task question --attention none --lr 0.001 --hs 256 --parse_strategy right_branching`
- `python seq2seq.py --encoder Tree --decoder Tree --task question --attention none --lr 0.001 --hs 256`
- `python seq2seq.py --encoder GRU --decoder GRU --task tense --attention none --lr 0.001 --hs 256`
- `python seq2seq.py --encoder GRU --decoder GRU --task tense_bracket --attention none --lr 0.001 --hs 256 --patience 6`
- `python seq2seq.py --encoder Tree --decoder Tree --task tense --attention none --lr 0.001 --hs 256 --parse_strategy right_branching`
- `python seq2seq.py --encoder Tree --decoder Tree --task tense --attention none --lr 0.001 --hs 256`

Run the following evaluations once each:
- `python test_question.py --encoder GRU --decoder GRU --task question --attention none --lr 0.001 --hs 256 > GRU_GRU_question_none_0.001_256.results`
- `python test_question.py --encoder GRU --decoder GRU --task question_bracket --attention none --lr 0.001 --hs 256 --patience 6 > GRU_GRU_question_bracket_none_0.001_256.results`
- `python test_question.py --encoder Tree --decoder Tree --task question --attention none --lr 0.001 --hs 256 --parse_strategy right_branching > TreeRB_TreeRB_question_none_0.001_256.results`
- `python test_question.py --encoder Tree --decoder Tree --task question --attention none --lr 0.001 --hs 256 > Tree_Tree_question_none_0.001_256.results`
- `python test_tense.py --encoder GRU --decoder GRU --task tense --attention none --lr 0.001 --hs 256 > GRU_GRU_tense_none_0.001_256.results`
- `python test_tense.py --encoder GRU --decoder GRU --task tense_bracket --attention none --lr 0.001 --hs 256 --patience 6 > GRU_GRU_tense_bracket_none_0.001_256.results`
- `python test_tense.py --encoder Tree --decoder Tree --task tense --attention none --lr 0.001 --hs 256 --parse_strategy right_branching > TreeRB_TreeRB_tense_none_0.001_256.results`
- `python test_tense.py --encoder Tree --decoder Tree --task tense --attention none --lr 0.001 --hs 256 > Tree_Tree_question_none_0.001_256.results`


### Multitask learning (Section 8 / Figure 10)
Run the following training steps 100 times each:
- `python seq2seq.py --encoder GRU --decoder GRU --task question --attention none --lr 0.001 --hs 256`
- `python seq2seq.py --encoder GRU --decoder GRU --task tense --attention none --lr 0.001 --hs 256`
- `python seq2seq.py --encoder GRU --decoder GRU --task question_tense_subject --attention none --lr 0.001 --hs 256`
- `python seq2seq.py --encoder GRU --decoder GRU --task question_main_tense --attention none --lr 0.001 --hs 256`
- `python seq2seq.py --encoder GRU --decoder GRU --task question_tense_aux_subject --attention none --lr 0.001 --hs 256`
- `python seq2seq.py --encoder GRU --decoder GRU --task question_main_tense_aux --attention none --lr 0.001 --hs 256`

Run the following evaluations once each:
- `python test_question.py --encoder GRU --decoder GRU --task question --attention none --lr 0.001 --hs 256 > GRU_GRU_question_none_0.001_256.results`
- `python test_tense.py --encoder GRU --decoder GRU --task tense --attention none --lr 0.001 --hs 256 > GRU_GRU_tense_main_none_0.001_256.results`
- `python test_question.py --encoder GRU --decoder GRU --task question_tense_subject --attention none --lr 0.001 --hs 256 > GRU_GRU_question_tense_subject_none_0.001_256.results`
- `python test_tense.py --encoder GRU --decoder GRU --task question_main_tense --attention none --lr 0.001 --hs 256 > GRU_GRU_question_main_tense_none_0.001_256.results`
- `python test_question.py --encoder GRU --decoder GRU --task question_tense_aux_subject --attention none --lr 0.001 --hs 256 > GRU_GRU_question_tense_aux_subject_none_0.001_256.results`
- `python test_tense_aux.py --encoder GRU --decoder GRU --task question_main_tense_aux --attention none --lr 0.001 --hs 256 > GRU_GRU_question_main_tense_aux_none_0.001_256.results`

# CFG


