#!/bin/bash


# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
data=/data/kaldi/egs/librispeech/s5/data

# base url for downloads.
data_url=www.openslr.org/resources/12
lm_url=www.openslr.org/resources/11

. ./cmd.sh
. ./path.sh

# you might not want to do this for interactive shells.
set -e

# download the data.  Note: we're using the 100 hour setup for
# now; later in the script we'll download more and use it to train neural
# nets.
#wsd1# for part in dev-clean test-clean dev-other test-other train-clean-100; do
#wsd1#   local/download_and_untar.sh $data $data_url $part
#wsd1# done
#wsd1# 
#wsd1# # download the LM resources
#wsd1# local/download_lm.sh $lm_url data/local/lm
#wsd1# 
#wsd1# # format the data as Kaldi data directories
#wsd1# for part in dev-clean test-clean dev-other test-other train-clean-100; do
#wsd1#   # use underscore-separated names in data directories.
#wsd1#   local/data_prep.sh $data/LibriSpeech/$part data/$(echo $part | sed s/-/_/g)
#wsd1# done
#wsd1# 
#wsd1# ## Optional text corpus normalization and LM training
#wsd1# ## These scripts are here primarily as a documentation of the process that has been
#wsd1# ## used to build the LM. Most users of this recipe will NOT need/want to run
#wsd1# ## this step. The pre-built language models and the pronunciation lexicon, as
#wsd1# ## well as some intermediate data(e.g. the normalized text used for LM training),
#wsd1# ## are available for download at http://www.openslr.org/11/
#wsd1# #local/lm/train_lm.sh $LM_CORPUS_ROOT \
#wsd1# #  data/local/lm/norm/tmp data/local/lm/norm/norm_texts data/local/lm
#wsd1# 
#wsd1# ## Optional G2P training scripts.
#wsd1# ## As the LM training scripts above, this script is intended primarily to
#wsd1# ## document our G2P model creation process
#wsd1# #local/g2p/train_g2p.sh data/local/dict/cmudict data/local/lm
#wsd1# 
#wsd1# # when "--stage 3" option is used below we skip the G2P steps, and use the
#wsd1# # lexicon we have already downloaded from openslr.org/11/
#wsd1# local/prepare_dict.sh --stage 3 --nj 30 --cmd "$train_cmd" \
#wsd1#    data/local/lm data/local/lm data/local/dict_nosp
#wsd1# 
#wsd1# utils/prepare_lang.sh data/local/dict_nosp \
#wsd1#   "<UNK>" data/local/lang_tmp_nosp data/lang_nosp
#wsd1# 
#wsd1# local/format_lms.sh --src-dir data/lang_nosp data/local/lm
#wsd1# 
#wsd1# # Create ConstArpaLm format language model for full 3-gram and 4-gram LMs
#wsd1# utils/build_const_arpa_lm.sh data/local/lm/lm_tglarge.arpa.gz \
#wsd1#   data/lang_nosp data/lang_nosp_test_tglarge
#wsd1# utils/build_const_arpa_lm.sh data/local/lm/lm_fglarge.arpa.gz \
#wsd1#   data/lang_nosp data/lang_nosp_test_fglarge
#wsd1# 
#wsd1# mfccdir=mfcc
#wsd1# # spread the mfccs over various machines, as this data-set is quite large.
#wsd1# if [[  $(hostname -f) ==  *.clsp.jhu.edu ]]; then
#wsd1#   mfcc=$(basename mfccdir) # in case was absolute pathname (unlikely), get basename.
#wsd1#   utils/create_split_dir.pl /export/b{02,11,12,13}/$USER/kaldi-data/egs/librispeech/s5/$mfcc/storage \
#wsd1#     $mfccdir/storage
#wsd1# fi
#wsd1# 
#wsd1# 
#wsd1# for part in dev_clean test_clean dev_other test_other train_clean_100; do
#wsd1#   steps/make_mfcc.sh --cmd "$train_cmd" --nj 40 data/$part exp/make_mfcc/$part $mfccdir
#wsd1#   steps/compute_cmvn_stats.sh data/$part exp/make_mfcc/$part $mfccdir
#wsd1# done
#wsd1# 
#wsd1# # Make some small data subsets for early system-build stages.  Note, there are 29k
#wsd1# # utterances in the train_clean_100 directory which has 100 hours of data.
#wsd1# # For the monophone stages we select the shortest utterances, which should make it
#wsd1# # easier to align the data from a flat start.
#wsd1# 
#wsd1# utils/subset_data_dir.sh --shortest data/train_clean_100 2000 data/train_2kshort
#wsd1# utils/subset_data_dir.sh data/train_clean_100 5000 data/train_5k
#wsd1# utils/subset_data_dir.sh data/train_clean_100 10000 data/train_10k
#wsd1# 
#wsd1# # train a monophone system
#wsd1# steps/train_mono.sh --boost-silence 1.25 --nj 20 --cmd "$train_cmd" \
#wsd1#   data/train_2kshort data/lang_nosp exp/mono
#wsd1# 
#wsd1# # decode using the monophone model
#wsd1# (
#wsd1#   utils/mkgraph.sh data/lang_nosp_test_tgsmall \
#wsd1#     exp/mono exp/mono/graph_nosp_tgsmall
#wsd1#   for test in test_clean test_other dev_clean dev_other; do
#wsd1#     steps/decode.sh --nj 20 --cmd "$decode_cmd" exp/mono/graph_nosp_tgsmall \
#wsd1#       data/$test exp/mono/decode_nosp_tgsmall_$test
#wsd1#   done
#wsd1# )&
#wsd1# 
#wsd1# steps/align_si.sh --boost-silence 1.25 --nj 10 --cmd "$train_cmd" \
#wsd1#   data/train_5k data/lang_nosp exp/mono exp/mono_ali_5k
#wsd1# 
#wsd1# # train a first delta + delta-delta triphone system on a subset of 5000 utterances
#wsd1# steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
#wsd1#     2000 10000 data/train_5k data/lang_nosp exp/mono_ali_5k exp/tri1
#wsd1# 
#wsd1# # decode using the tri1 model
#wsd1# (
#wsd1#   utils/mkgraph.sh data/lang_nosp_test_tgsmall \
#wsd1#     exp/tri1 exp/tri1/graph_nosp_tgsmall
#wsd1#   for test in test_clean test_other dev_clean dev_other; do
#wsd1#     steps/decode.sh --nj 20 --cmd "$decode_cmd" exp/tri1/graph_nosp_tgsmall \
#wsd1#       data/$test exp/tri1/decode_nosp_tgsmall_$test
#wsd1#     steps/lmrescore.sh --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tgmed} \
#wsd1#       data/$test exp/tri1/decode_nosp_{tgsmall,tgmed}_$test
#wsd1#     steps/lmrescore_const_arpa.sh \
#wsd1#       --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tglarge} \
#wsd1#       data/$test exp/tri1/decode_nosp_{tgsmall,tglarge}_$test
#wsd1#   done
#wsd1# )&
#wsd1# 
#wsd1# steps/align_si.sh --nj 10 --cmd "$train_cmd" \
#wsd1#   data/train_10k data/lang_nosp exp/tri1 exp/tri1_ali_10k
#wsd1# 
#wsd1# 
#wsd1# # train an LDA+MLLT system.
#wsd1# steps/train_lda_mllt.sh --cmd "$train_cmd" \
#wsd1#    --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
#wsd1#    data/train_10k data/lang_nosp exp/tri1_ali_10k exp/tri2b
#wsd1# 
#wsd1# # decode using the LDA+MLLT model
#wsd1# (
#wsd1#   utils/mkgraph.sh data/lang_nosp_test_tgsmall \
#wsd1#     exp/tri2b exp/tri2b/graph_nosp_tgsmall
#wsd1#   for test in test_clean test_other dev_clean dev_other; do
#wsd1#     steps/decode.sh --nj 20 --cmd "$decode_cmd" exp/tri2b/graph_nosp_tgsmall \
#wsd1#       data/$test exp/tri2b/decode_nosp_tgsmall_$test
#wsd1#     steps/lmrescore.sh --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tgmed} \
#wsd1#       data/$test exp/tri2b/decode_nosp_{tgsmall,tgmed}_$test
#wsd1#     steps/lmrescore_const_arpa.sh \
#wsd1#       --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tglarge} \
#wsd1#       data/$test exp/tri2b/decode_nosp_{tgsmall,tglarge}_$test
#wsd1#   done
#wsd1# )&
#wsd1# 
#wsd1# # Align a 10k utts subset using the tri2b model
#wsd1# steps/align_si.sh  --nj 10 --cmd "$train_cmd" --use-graphs true \
#wsd1#   data/train_10k data/lang_nosp exp/tri2b exp/tri2b_ali_10k
#wsd1# 
#wsd1# # Train tri3b, which is LDA+MLLT+SAT on 10k utts
#wsd1# steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
#wsd1#   data/train_10k data/lang_nosp exp/tri2b_ali_10k exp/tri3b
#wsd1# 
#wsd1# # decode using the tri3b model
#wsd1# (
#wsd1#   utils/mkgraph.sh data/lang_nosp_test_tgsmall \
#wsd1#     exp/tri3b exp/tri3b/graph_nosp_tgsmall
#wsd1#   for test in test_clean test_other dev_clean dev_other; do
#wsd1#     steps/decode_fmllr.sh --nj 20 --cmd "$decode_cmd" \
#wsd1#       exp/tri3b/graph_nosp_tgsmall data/$test \
#wsd1#       exp/tri3b/decode_nosp_tgsmall_$test
#wsd1#     steps/lmrescore.sh --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tgmed} \
#wsd1#       data/$test exp/tri3b/decode_nosp_{tgsmall,tgmed}_$test
#wsd1#     steps/lmrescore_const_arpa.sh \
#wsd1#       --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tglarge} \
#wsd1#       data/$test exp/tri3b/decode_nosp_{tgsmall,tglarge}_$test
#wsd1#   done
#wsd1# )&
#wsd1# 
#wsd1# # align the entire train_clean_100 subset using the tri3b model
#wsd1# steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" \
#wsd1#   data/train_clean_100 data/lang_nosp \
#wsd1#   exp/tri3b exp/tri3b_ali_clean_100
#wsd1# 
#wsd1# # train another LDA+MLLT+SAT system on the entire 100 hour subset
#wsd1# steps/train_sat.sh  --cmd "$train_cmd" 4200 40000 \
#wsd1#   data/train_clean_100 data/lang_nosp \
#wsd1#   exp/tri3b_ali_clean_100 exp/tri4b
#wsd1# 
#wsd1# # decode using the tri4b model
#wsd1# (
#wsd1#   utils/mkgraph.sh data/lang_nosp_test_tgsmall \
#wsd1#     exp/tri4b exp/tri4b/graph_nosp_tgsmall
#wsd1#   for test in test_clean test_other dev_clean dev_other; do
#wsd1#     steps/decode_fmllr.sh --nj 20 --cmd "$decode_cmd" \
#wsd1#       exp/tri4b/graph_nosp_tgsmall data/$test \
#wsd1#       exp/tri4b/decode_nosp_tgsmall_$test
#wsd1#     steps/lmrescore.sh --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tgmed} \
#wsd1#       data/$test exp/tri4b/decode_nosp_{tgsmall,tgmed}_$test
#wsd1#     steps/lmrescore_const_arpa.sh \
#wsd1#       --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tglarge} \
#wsd1#       data/$test exp/tri4b/decode_nosp_{tgsmall,tglarge}_$test
#wsd1#     steps/lmrescore_const_arpa.sh \
#wsd1#       --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,fglarge} \
#wsd1#       data/$test exp/tri4b/decode_nosp_{tgsmall,fglarge}_$test
#wsd1#   done
#wsd1# )&
#wsd1# 
#wsd1# # Now we compute the pronunciation and silence probabilities from training data,
#wsd1# # and re-create the lang directory.
#wsd1# steps/get_prons.sh --cmd "$train_cmd" \
#wsd1#   data/train_clean_100 data/lang_nosp exp/tri4b
#wsd1# utils/dict_dir_add_pronprobs.sh --max-normalize true \
#wsd1#   data/local/dict_nosp \
#wsd1#   exp/tri4b/pron_counts_nowb.txt exp/tri4b/sil_counts_nowb.txt \
#wsd1#   exp/tri4b/pron_bigram_counts_nowb.txt data/local/dict
#wsd1# 
#wsd1# utils/prepare_lang.sh data/local/dict \
#wsd1#   "<UNK>" data/local/lang_tmp data/lang
#wsd1# local/format_lms.sh --src-dir data/lang data/local/lm
#wsd1# 
#wsd1# utils/build_const_arpa_lm.sh \
#wsd1#   data/local/lm/lm_tglarge.arpa.gz data/lang data/lang_test_tglarge
#wsd1# utils/build_const_arpa_lm.sh \
#wsd1#   data/local/lm/lm_fglarge.arpa.gz data/lang data/lang_test_fglarge
#wsd1# 
#wsd1# # decode using the tri4b model with pronunciation and silence probabilities
#wsd1# (
#wsd1#   utils/mkgraph.sh \
#wsd1#     data/lang_test_tgsmall exp/tri4b exp/tri4b/graph_tgsmall
#wsd1#   for test in test_clean test_other dev_clean dev_other; do
#wsd1#     steps/decode_fmllr.sh --nj 20 --cmd "$decode_cmd" \
#wsd1#       exp/tri4b/graph_tgsmall data/$test \
#wsd1#       exp/tri4b/decode_tgsmall_$test
#wsd1#     steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
#wsd1#       data/$test exp/tri4b/decode_{tgsmall,tgmed}_$test
#wsd1#     steps/lmrescore_const_arpa.sh \
#wsd1#       --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
#wsd1#       data/$test exp/tri4b/decode_{tgsmall,tglarge}_$test
#wsd1#     steps/lmrescore_const_arpa.sh \
#wsd1#       --cmd "$decode_cmd" data/lang_test_{tgsmall,fglarge} \
#wsd1#       data/$test exp/tri4b/decode_{tgsmall,fglarge}_$test
#wsd1#   done
#wsd1# )&
#wsd1# 
#wsd1# # align train_clean_100 using the tri4b model
#wsd1# steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
#wsd1#   data/train_clean_100 data/lang exp/tri4b exp/tri4b_ali_clean_100
#wsd1# 
#wsd1# # if you want at this point you can train and test NN model(s) on the 100 hour
#wsd1# # subset
#wsd1# local/nnet2/run_5a_clean_100.sh

local/download_and_untar.sh $data $data_url train-clean-360

# now add the "clean-360" subset to the mix ...
local/data_prep.sh \
  $data/LibriSpeech/train-clean-360 data/train_clean_360
steps/make_mfcc.sh --cmd "$train_cmd" --nj 40 data/train_clean_360 \
  exp/make_mfcc/train_clean_360 $mfccdir
steps/compute_cmvn_stats.sh \
  data/train_clean_360 exp/make_mfcc/train_clean_360 $mfccdir

# ... and then combine the two sets into a 460 hour one
utils/combine_data.sh \
  data/train_clean_460 data/train_clean_100 data/train_clean_360

# align the new, combined set, using the tri4b model
steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
  data/train_clean_460 data/lang exp/tri4b exp/tri4b_ali_clean_460

# create a larger SAT model, trained on the 460 hours of data.
steps/train_sat.sh  --cmd "$train_cmd" 5000 100000 \
  data/train_clean_460 data/lang exp/tri4b_ali_clean_460 exp/tri5b

# decode using the tri5b model
(
  utils/mkgraph.sh data/lang_test_tgsmall \
    exp/tri5b exp/tri5b/graph_tgsmall
  for test in test_clean test_other dev_clean dev_other; do
    steps/decode_fmllr.sh --nj 20 --cmd "$decode_cmd" \
      exp/tri5b/graph_tgsmall data/$test \
      exp/tri5b/decode_tgsmall_$test
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
      data/$test exp/tri5b/decode_{tgsmall,tgmed}_$test
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
      data/$test exp/tri5b/decode_{tgsmall,tglarge}_$test
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_test_{tgsmall,fglarge} \
      data/$test exp/tri5b/decode_{tgsmall,fglarge}_$test
  done
)&

# train a NN model on the 460 hour set
local/nnet2/run_6a_clean_460.sh

local/download_and_untar.sh $data $data_url train-other-500

# prepare the 500 hour subset.
local/data_prep.sh \
  $data/LibriSpeech/train-other-500 data/train_other_500
steps/make_mfcc.sh --cmd "$train_cmd" --nj 40 data/train_other_500 \
  exp/make_mfcc/train_other_500 $mfccdir
steps/compute_cmvn_stats.sh \
  data/train_other_500 exp/make_mfcc/train_other_500 $mfccdir

# combine all the data
utils/combine_data.sh \
  data/train_960 data/train_clean_460 data/train_other_500

steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
  data/train_960 data/lang exp/tri5b exp/tri5b_ali_960

# train a SAT model on the 960 hour mixed data.  Use the train_quick.sh script
# as it is faster.
steps/train_quick.sh --cmd "$train_cmd" \
  7000 150000 data/train_960 data/lang exp/tri5b_ali_960 exp/tri6b

# decode using the tri6b model
(
  utils/mkgraph.sh data/lang_test_tgsmall \
    exp/tri6b exp/tri6b/graph_tgsmall
  for test in test_clean test_other dev_clean dev_other; do
    steps/decode_fmllr.sh --nj 20 --cmd "$decode_cmd" \
      exp/tri6b/graph_tgsmall data/$test exp/tri6b/decode_tgsmall_$test
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
      data/$test exp/tri6b/decode_{tgsmall,tgmed}_$test
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
      data/$test exp/tri6b/decode_{tgsmall,tglarge}_$test
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_test_{tgsmall,fglarge} \
      data/$test exp/tri6b/decode_{tgsmall,fglarge}_$test
  done
)&

# this does some data-cleaning. The cleaned data should be useful when we add
# the neural net and chain systems.
#wsd1# local/run_cleanup_segmentation.sh

# steps/cleanup/debug_lexicon.sh --remove-stress true  --nj 200 --cmd "$train_cmd" data/train_clean_100 \
#    data/lang exp/tri6b data/local/dict/lexicon.txt exp/debug_lexicon_100h

# #Perform rescoring of tri6b be means of faster-rnnlm
# #Attention: with default settings requires 4 GB of memory per rescoring job, so commenting this out by default
# wait && local/run_rnnlm.sh \
#     --rnnlm-ver "faster-rnnlm" \
#     --rnnlm-options "-hidden 150 -direct 1000 -direct-order 5" \
#     --rnnlm-tag "h150-me5-1000" $data data/local/lm

# #Perform rescoring of tri6b be means of faster-rnnlm using Noise contrastive estimation
# #Note, that could be extremely slow without CUDA
# #We use smaller direct layer size so that it could be stored in GPU memory (~2Gb)
# #Suprisingly, bottleneck here is validation rather then learning
# #Therefore you can use smaller validation dataset to speed up training
# wait && local/run_rnnlm.sh \
#     --rnnlm-ver "faster-rnnlm" \
#     --rnnlm-options "-hidden 150 -direct 400 -direct-order 3 --nce 20" \
#     --rnnlm-tag "h150-me3-400-nce20" $data data/local/lm


# train nnet3 tdnn models on the entire data with data-cleaning (xent and chain)
#wsd1# local/chain/run_tdnn.sh # set "--stage 11" if you have already run local/nnet3/run_tdnn.sh

# The nnet3 TDNN recipe:
# local/nnet3/run_tdnn.sh # set "--stage 11" if you have already run local/chain/run_tdnn.sh

# # train models on cleaned-up data
# # we've found that this isn't helpful-- see the comments in local/run_data_cleaning.sh
# local/run_data_cleaning.sh

# # The following is the current online-nnet2 recipe, with "multi-splice".
# local/online/run_nnet2_ms.sh

# # The following is the discriminative-training continuation of the above.
# local/online/run_nnet2_ms_disc.sh

# ## The following is an older version of the online-nnet2 recipe, without "multi-splice".  It's faster
# ## to train but slightly worse.
# # local/online/run_nnet2.sh

# Wait for decodings in the background
wait
