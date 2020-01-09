# rnn-hierarchical-biases
Code for "Does syntax need to grow on trees? Sources of hierarchical inductive bias in sequence to sequence networks." The paper can be found at [LINK NOT YET AVAILABLE], with an earlier paper available [here](https://arxiv.org/pdf/1802.09091.pdf) (the experiments in the earlier paper are mostly a subset of the ones in the newer paper, but the older one does contain some analyses that are not in the newer one). There is also a website containing detailed results at [http://rtmccoy.com/rnn_hierarchical_biases.html](http://rtmccoy.com/rnn_hierarchical_biases.html).


Contents of this README:
- [Dependencies](#dependencies)
- [Basic description of the code](#description)
- [Understanding the test output](#understanding-output)
- [Example experiment](#example)
- [Data](#data)
- [How to replicate the experiments in the paper](#replication) (section and figure numbers refer to the paper):
    * [Variants on sequential RNNs (Section 3.3 / Figure 4)](#seq)
    * [Squashing experiments (Section 3.4 / Figure 5)](#squashing)
    * [Ordered Neurons model for question formation (Section 4.1)](#onlstm)
    * [Tree-GRUs for question formation (Section 4.2 / Figure 7)](#tree)
    * [Tense reinflection (Section 5.1 / Figure 8)](#tense)
    * [Unambiguous training sets (Section 6)](#unambiguous)
    * [Tree structure vs. tree information (Section 7 / Figure 9)](#structureinfo)
    * [Multitask learning (Section 8 / Figure 10)](#multitask)
- [Citing this code](#citing)

# [Dependencies](#dependencies)
We ran this with PyTorch version 0.4.0, but other versions may well work. It should run on either a GPU or CPU.

# [Basic description of the code](#description)

Each model consists of two sub-models, the encoder and the decoder. An experiment is split into two parts: First, the script `seq2seq.py` trains the encoder and decoder and saves their weights in a directory that is automatically generated for that experiment. Second, the test scripts (`test_question.py` for evaluating question formation, `test_tense.py` for evaluating tense reinflection, and `test_tense_aux.py` for evaluating tense reinflection with auxiliaries included) evaluate the trained models. If you want to run multiple random restarts of a given model, simply launch `seq2seq.py` multiple times; the code will automatically generate a different directory for each restart. You will still only need to run the test script once, however, as it will compile together all initializations that have been run for the provided hyperparameters.

Both the training and testing scripts take the same set of options:
- `--encoder`: Encoder type. Options:
    * `SRN`: Simple Recurrent Network.
    * `GRU`: Gated Recurrent Unit.
    * `LSTM`: Long Short-Term Memory unit.
    * `SquashedLSTM`: Squashed Long Short-Term Memory unit (see section 3.4 of the paper).
    * `UnsquashedGRU`: Unsquashed Gated Recurrent Unit (see section 3.4 of the paper).
    * `ONLSTM`: Ordered Neurons LSTM.
    * `Tree`: Tree-based GRU.
- `--decoder`: Decoder type. Options:
    * `SRN`: Simple Recurrent Network.
    * `GRU`: Gated Recurrent Unit.
    * `LSTM`: Long Short-Term Memory unit.
    * `SquashedLSTM`: Squashed Long Short-Term Memory unit (see section 3.4 of the paper).
    * `UnsquashedGRU`: Unsquashed Gated Recurrent Unit (see section 3.4 of the paper).
    * `ONLSTM`: Ordered Neurons LSTM.
    * `Tree`: Tree-based GRU.
- `--task`: Task. The options that can be used here are the datasets listed in the "Data" section of this README (e.g., `question`, `tense`, `question_tense_aux_subject`, etc.)
- `--attention`: Type of attention. Options:
    * `none`: No attention.
    * `location`: Location-based attention.
    * `content`: Content-based attention.
- `--lr`: Learning rate. Can be any float value. Most of our experiments were run with a learning rate of 0.001.
- `--hs`: Hidden size. Can be any integer value. Most of our experiments were run with a hidden size of 256.
- `--seed`: Random seed. If not provided manually, the code will internally generate a random seed.
- `--parse_strategy`: If using a tree-based model, the type of parse that will be used. Options:
    * `correct`: The correct parses (default option).
    * `right_branching`: Uniformly right-branching trees.
- `--patience`: Number of evaluation steps without improvement to go through before early stopping. Can take any integer value; default value is 3.

The section of this README entitled "How to replicate the experiments in the paper" gives the specific commands you would need to run to replicate the experiments reported in our paper.

In case you want to run your own experiments with different combinations of arguments, note that the following combinations are currently unsupported:
- Using a Tree encoder or decoder with the tasks `tense_aux`, `tense_aux_subject`, `question_bracket`, `tense_bracket`, 'question_main_tense_aux`, and `question_tense_aux_subject`.
- Using attention (whether location-based or content-based) with a Tree encoder or decoder.

# [Understanding the test output](#understanding-output)

The evaluation scripts (`test_question.py` for evaluating question formation, `test_tense.py` for evaluating tense reinflection, and `test_tense_aux.py` for evaluating tense reinflection with auxiliaries included) need to only be run once for a given type of model; a single run of the script will evaluate all instances that have been trained for the model with the specified hyperparameters. The output of these scripts first gives example outputs for each trained model; for each model there are first some example outputs from the test set, under the heading "Test set example outputs," followed by some example outputs from the generalization set, under the heading "Generalization set example outputs." Each example has 3 lines: first the input, then the target output, then the model's predicted output.

At the bottom of the document are the metrics used to evaluate the models. For each metric, there is first a list of each model's result on that metric (e.g., if you ran 100 instances of the model, each of those lists will have length 100). Under the list are then the mean and median values for that list. The metrics given are:

- Question formation metrics:
    * Test full-sentence accuracy: Proportion of test set examples for which the output was exactly correct.
    * Test full-sentence POS accuracy list: Proportion of test set examples for which the output words all had the correct part-of-speech tag (but might not be the correct member of that part of speech).
    * Gen first word accuracy: Proportion of generalization set examples for which the first word of the output was the correct word (i.e., the main auxiliary of the input)
    * Gen proportion of outputs where the first word was the first auxiliary: Self-explanatory
    * Gen proportion of outputs where the first word was an auxiliary not in the input: Self-explanatory
    * Gen proportion of outputs where the first word was not an auxiliary: Self-explanatory
    * Gen full sentence accuracy: Proportion of generalization set examples for which the output was exactly correct
    * Gen full sentence POS accuracy: Proportion of generalization set examples for which the output had exactly the correct sequence of parts of speech
    * Output categorizations: Categorizing the outputs based on which auxiliary was placed at the front and which was deleted from within the sentence. Note that, for the generalization set, the second auxiliary is always also the main auxiliary. The categories are:
        - d1p1: First auxiliary deleted, first auxiliary preposed. Example: my walrus that does laugh doesn't giggle. -> does my walrus that laugh doesn't giggle?
        - d1p2: First auxiliary deleted, second auxiliary preposed. Example: my walrus that does laugh doesn't giggle. -> doesn't my walrus that laugh doesn't giggle?
        - d1po: First auxiliary deleted, auxiliary not in the input preposed. Example: my walrus that does laugh doesn't giggle. -> do my walrus that laugh doesn't giggle?
        - d2p1: Second auxiliary deleted, first auxiliary preposed. Example: my walrus that does laugh doesn't giggle. -> does my walrus that does laugh giggle?
        - d2p2: Second auxiliary deleted, second auxiliary preposed (correct output). Example: my walrus that does laugh doesn't giggle. -> doesn't my walrus that does laugh giggle?
        - d2po: Second auxiliary deleted, auxiliary not in the input preposed. Example: my walrus that does laugh doesn't giggle. -> don't my walrus that does laugh giggle?
        - dnp1: No auxiliary deleted, first auxiliary preposed. Example: my walrus that does laugh doesn't giggle. -> does my walrus that does laugh doesn't giggle?
        - dnp2: No auxiliary deleted, second auxiliary preposed. Example: my walrus that does laugh doesn't giggle. -> doesn't my walrus that does laugh doesn't giggle?
        - dnpo: No auxiliary deleted, auxiliary not in the input preposed. Example: my walrus that does laugh doesn't giggle. -> don't my walrus that does laugh doesn't giggle?
        - other: Output does not fit into any of the above categories.
    * ORC: The first-word accuracy across examples in the generalization set for which the relative clause modifying the subject is an object relative clause (that is, a relative clause in which the verb is transitive and the element moved out of the relative clause is the object of the verb; e.g., "that the walrus does visit").
    * SRC_t: The first-word accuracy across examples in the generalization set for which the relative clause modifying the subject is a transitive subject relative clause (that is, a relative clause in which the verb is transitive and the element moved out of the relative clause is the subject of the verb; e.g., "that does visit the walrus").
    * SRC_i: The first-word accuracy across examples in the generalization set for which the relative clause modifying the subject is an intransitive subject relative clause (that is, a relative clause in which the verb is intransitive and the element moved out of the relative clause is the subject of the verb; e.g., "that does giggle").
- Tense reinflection metrics:
    * Test full-sentence accuracy: Proportion of test set examples for which the output was exactly correct.
    * Test full-sentence POS accuracy list: Proportion of test set examples for which the output words all had the correct part-of-speech tag (but might not be the correct member of that part of speech).
    * Gen full sentence accuracy: Proportion of generalization set examples for which the output was exactly correct
    * Gen full sentence POS accuracy: Proportion of generalization set examples for which the output was exactly correct sequence of parts of speec
    * Gen proportion of full-sentence outputs that follow agree-recent: Self-explanatory. Example: my walrus by the yaks giggled. -< my walrus by the yaks giggle.
    * Gen proportion of outputs that have the correct main verb: Self-explanatory. Note: This presupposes that the output has the correct sequence of part-of-speech tags. If it doesn't, the sentence will not be counted as having the correct main verb or the incorrect main verb. 
    * Gen proportion of outputs that have the main verb predicted by agree-recent: Self-explanatory. Note: This presupposes that the output has the correct sequence of part-of-speech tags. If it doesn't, the sentence will not be counted as having the correct main verb or the incorrect main verb.
    * Gen proportion of outputs that have the correct number for the main verb: Self-explanatory. Note: This presupposes that the output has the correct sequence of part-of-speech tags. If it doesn't, the sentence will not be counted as having the correct main verb number or the incorrect main verb number.
    * Gen proportion of outputs that have the incorrect number for the main verb: Self-explanatory. Note: This presupposes that the output has the correct sequence of part-of-speech tags. If it doesn't, the sentence will not be counted as having the correct main verb number or the incorrect main verb number.


# [Example experiment](#example)

This repo contains one example, where we have trained and evaluated 3 instances of a GRU with location-based attention trained to perform question formation. These three instances were trained by running the following commands:

```
python seq2seq.py --encoder GRU --decoder GRU --task question --attention location --lr 0.001 --hs 256
python seq2seq.py --encoder GRU --decoder GRU --task question --attention location --lr 0.001 --hs 256
python seq2seq.py --encoder GRU --decoder GRU --task question --attention location --lr 0.001 --hs 256

```

Notice that this is just the same command repeated 3 times; `seq2seq.py` will automatically create 3 separate folders for these 3 runs. These folders are named `question_GRU_GRU_location_0.001_256_0`, `question_GRU_GRU_location_0.001_256_1`, and `question_GRU_GRU_location_0.001_256_0`. Each one contains the saved weights of the trained encoder, named `question.encoder.0.0.0`, and the saved weights of the decoder, named `question.decoder.0.0.0` (more specifically, the weights that were saved were the weights from the evaluation step that achieved the highest accuracy on the development set).

All 3 of these trained models were then evaluated together using the following command:

```
python test_question.py --encoder GRU --decoder GRU --task question --attention location --lr 0.001 --hs 256 > GRU_GRU_question_location_0.001_256.results
``` 

The evaluation results are in `GRU_GRU_question_location_0.001_256.results`. This document first gives example outputs for each of the 3 models: For each model, there are first some example outputs for the test set under the heading "Test set example outputs" (most of these are exactly correct) followed by some example outputs for the generalization set under the heading "Generalization set example outputs" (most of these are incorrect in various ways). Each of these examples has the input on the first line, followed by the target output on the second line, followed by the model's output on the third line. 

After these examples, the bottom of the document contains several evaluation metrics for all 3 models. For each metric, there is first a list of each model's value for that metric; since there are 3 instances of the model in this case, each of these lists is 3 elements long. Below that are then the mean and median values for that metric across models.

# [Data](#data)

All of the datasets are in `data/`. Each task is split across 4 files, namely a training set, a development set, a test set, and a generalization set, marked by the suffixes `.train`, `.dev`, `.test`, and `.gen`. Each task is indicated by a prefix:

- Basic datasets:
    * `question`: The basic question formation dataset
    * `tense`: The basic tense reinflection dataset
    * `tense_aux`: The basic tense reinflection dataset, but with auxiliaries before the verbs instead of inflected verbs (e.g., *does swim* instead of *swims*)
- Unambiguous datasets:
    * `question_main`: Question formation that is unambiguously governed by move-main.
    * `question_first`: Question formation that is unambiguously governed by move-first.
    * `tense_subject`: Tense reinflection that is unambiguously governed by agree-subject.
    * `tense_recent`: Tense reinflection that is unambiguously governed by agree-recent.
    * `tense_aux_subject`: Tense reinflection that is unambiguously governed by agree-subject and that has auxiliaries before the verbs instead of inflected verbs.
- Datasets with brackets:
    * `question_bracket`: Question formation with brackets in the input and output.
    * `tense_bracket`: Tense reinflection with brackets in the input and output.
- Multitask datasets:
    * `question_main_tense`: Question formation that is unambiguously governed by move-main, plus tense reinflection that is ambiguous between agree-subject and agree-recent.
    * `question_main_tense_aux`: Question formation that is unambiguously governed by move-main, plus tense reinflection that is ambiguous between agre
e-subject and agree-recent and that (unlike the basic tense reinflection, but like the basic question formation) has auxiliaries in the sentences.
    * `question_tense_subject`: Question formation that is ambiguous between move-main and move-first, plus tense reinflection that is unambiguously governed by agree-subject.
    * `question_tense_aux_subject`: Question formation that is ambiguous between move-main and move-first, plus tense reinflection that is unambiguously governed by agree-subject and that (unlike the basic tense reinflection, but like the basic question formation) has auxiliaries in the sentences.

These datasets were generated from context free grammars; the grammars used for the basic question formation and tense reinflection datasets are given in `cfgs/`. Each file is a PCFG, with one rule per line, in the form probability-tab-left_hand_side-tab-right_hand_side. Sentences from these basic grammars were then post-processed to give the datasets in `data/` (where the post-processing filtered the datasets to withhold the types of examples necessary to make the training sets ambiguous).



# [How to replicate the experiments in the paper](#replication)

The section numbers and figure numbers in the headings below refer to the paper.

### [Variants on sequential RNNs (Section 3.3 / Figure 4)](#seq)

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


### [Squashing experiments (Section 3.4 / Figure 5)](#squashing)

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

### [Ordered Neurons model for question formation (Section 4.1)](#onlstm)

Run this training step 100 times:
- `python seq2seq.py --encoder ONLSTM --decoder ONLSTM --task question --attention none --lr 0.001 --hs 256`

Run this evaluation step once:
- `python test_question.py --encoder ONLSTM --decoder ONLSTM --task question --attention none --lr 0.001 --hs 256 > ONLSTM_ONLSTM_question_none_0.001_256.results`

### [Tree-GRUs for question formation (Section 4.2 / Figure 7)](#tree)
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


### [Tense reinflection (Section 5.1 / Figure 8)](#tense)
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


### [Unambiguous training sets (Section 6)](#unambiguous)
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


### [Tree structure vs. tree information (Section 7 / Figure 9)](#structureinfo)
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


### [Multitask learning (Section 8 / Figure 10)](#multitask)
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


# [Citing this code](#citing)

If you use this code in your work, please cite the following paper ([bibtex](http://tommccoy1.github.io/bibtex/rnn-hierarchical-biases.html)):

R. Thomas McCoy, Robert Frank, and Tal Linzen. 2020. Does Syntax Need to Grow on Trees? Sources of Hierarchical Inductive Bias in Sequence-to-Sequence Networks. To appear in *Transactions of the Association for Computational Linguistics*.


*Questions? Comments? Email [tom.mccoy@jhu.edu](mailto:tom.mccoy@jhu.edu).*

