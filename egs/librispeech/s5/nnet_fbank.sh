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
#wsd# for part in dev-clean test-clean dev-other test-other train-clean-100; do
#wsd#   local/download_and_untar.sh $data $data_url $part
#wsd# done

# download the LM resources
#wsd# local/download_lm.sh $lm_url data/local/lm

# format the data as Kaldi data directories
#wsd# for part in dev-clean test-clean dev-other test-other train-clean-100; do
#wsd#   # use underscore-separated names in data directories.
#wsd#   local/data_prep.sh $data/LibriSpeech/$part data/$(echo $part | sed s/-/_/g)
#wsd# done

## Optional text corpus normalization and LM training
## These scripts are here primarily as a documentation of the process that has been
## used to build the LM. Most users of this recipe will NOT need/want to run
## this step. The pre-built language models and the pronunciation lexicon, as
## well as some intermediate data(e.g. the normalized text used for LM training),
## are available for download at http://www.openslr.org/11/
#local/lm/train_lm.sh $LM_CORPUS_ROOT \
#  data/local/lm/norm/tmp data/local/lm/norm/norm_texts data/local/lm

## Optional G2P training scripts.
## As the LM training scripts above, this script is intended primarily to
## document our G2P model creation process
#local/g2p/train_g2p.sh data/local/dict/cmudict data/local/lm

# when "--stage 3" option is used below we skip the G2P steps, and use the
# lexicon we have already downloaded from openslr.org/11/
#wsd# local/prepare_dict.sh --stage 3 --nj 30 --cmd "$train_cmd" \
#wsd#   data/local/lm data/local/lm data/local/dict_nosp

#wsd# utils/prepare_lang.sh data/local/dict_nosp \
#wsd#  "<UNK>" data/local/lang_tmp_nosp data/lang_nosp

#wsd# local/format_lms.sh --src-dir data/lang_nosp data/local/lm

# Create ConstArpaLm format language model for full 3-gram and 4-gram LMs
#wsd# utils/build_const_arpa_lm.sh data/local/lm/lm_tglarge.arpa.gz \
#wsd#   data/lang_nosp data/lang_nosp_test_tglarge
#wsd# utils/build_const_arpa_lm.sh data/local/lm/lm_fglarge.arpa.gz \
#wsd#   data/lang_nosp data/lang_nosp_test_fglarge

#wsd# mfccdir=mfcc
# spread the mfccs over various machines, as this data-set is quite large.
#wsd# if [[  $(hostname -f) ==  *.clsp.jhu.edu ]]; then
#wsd#   mfcc=$(basename mfccdir) # in case was absolute pathname (unlikely), get basename.
#wsd#   utils/create_split_dir.pl /export/b{02,11,12,13}/$USER/kaldi-data/egs/librispeech/s5/$mfcc/storage \
#wsd#     $mfccdir/storage
#wsd# fi


#wsd# for part in dev_clean test_clean dev_other test_other train_clean_100; do
#wsd#   steps/make_mfcc.sh --cmd "$train_cmd" --nj 40 data/$part exp/make_mfcc/$part $mfccdir
#wsd#   steps/compute_cmvn_stats.sh data/$part exp/make_mfcc/$part $mfccdir
#wsd# done

#### <<wsd fbank> ############
#### build filterbank features

#wsd6# fbankdir=fbank
#wsd6# mkdir -p data-fbank
#wsd6# for part in dev_clean test_clean dev_other test_other train_10k train_clean_100; do
#wsd6#   cp -r data/$part data-fbank/$part
#wsd6#   steps/make_fbank.sh --cmd "$train_cmd" --nj 40 data-fbank/$part exp/make_fbank/$part $fbankdir
#wsd6# done
#wsd6# #### <wsd fbank>> ############

# Make some small data subsets for early system-build stages.  Note, there are 29k
# utterances in the train_clean_100 directory which has 100 hours of data.
# For the monophone stages we select the shortest utterances, which should make it
# easier to align the data from a flat start.

#wsd# utils/subset_data_dir.sh --shortest data/train_clean_100 2000 data/train_2kshort
#wsd# utils/subset_data_dir.sh data/train_clean_100 5000 data/train_5k
#wsd# utils/subset_data_dir.sh data/train_clean_100 10000 data/train_10k

# train a monophone system
#wsd# steps/train_mono.sh --boost-silence 1.25 --nj 20 --cmd "$train_cmd" \
#wsd#   data/train_2kshort data/lang_nosp exp/mono

# decode using the monophone model
#wsd# (
#wsd#   utils/mkgraph.sh data/lang_nosp_test_tgsmall \
#wsd#     exp/mono exp/mono/graph_nosp_tgsmall
#wsd#   for test in test_clean test_other dev_clean dev_other; do
#wsd#     steps/decode.sh --nj 20 --cmd "$decode_cmd" exp/mono/graph_nosp_tgsmall \
#wsd#       data/$test exp/mono/decode_nosp_tgsmall_$test
#wsd#   done
#wsd# )&

#wsd# steps/align_si.sh --boost-silence 1.25 --nj 10 --cmd "$train_cmd" \
#wsd#   data/train_5k data/lang_nosp exp/mono exp/mono_ali_5k

# train a first delta + delta-delta triphone system on a subset of 5000 utterances
#wsd# steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
#wsd#     2000 10000 data/train_5k data/lang_nosp exp/mono_ali_5k exp/tri1

# decode using the tri1 model
#wsd# (
#wsd#   utils/mkgraph.sh data/lang_nosp_test_tgsmall \
#wsd#     exp/tri1 exp/tri1/graph_nosp_tgsmall
#wsd#   for test in test_clean test_other dev_clean dev_other; do
#wsd#     steps/decode.sh --nj 20 --cmd "$decode_cmd" exp/tri1/graph_nosp_tgsmall \
#wsd#       data/$test exp/tri1/decode_nosp_tgsmall_$test
#wsd#     steps/lmrescore.sh --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tgmed} \
#wsd#       data/$test exp/tri1/decode_nosp_{tgsmall,tgmed}_$test
#wsd#     steps/lmrescore_const_arpa.sh \
#wsd#       --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tglarge} \
#wsd#       data/$test exp/tri1/decode_nosp_{tgsmall,tglarge}_$test
#wsd#   done
#wsd# )&

#wsd# steps/align_si.sh --nj 10 --cmd "$train_cmd" \
#wsd#   data/train_10k data/lang_nosp exp/tri1 exp/tri1_ali_10k


# train an LDA+MLLT system.
#wsd# steps/train_lda_mllt.sh --cmd "$train_cmd" \
#wsd#    --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
#wsd#    data/train_10k data/lang_nosp exp/tri1_ali_10k exp/tri2b

# decode using the LDA+MLLT model
#wsd# (
#wsd#   utils/mkgraph.sh data/lang_nosp_test_tgsmall \
#wsd#     exp/tri2b exp/tri2b/graph_nosp_tgsmall
#wsd#   for test in test_clean test_other dev_clean dev_other; do
#wsd#     steps/decode.sh --nj 20 --cmd "$decode_cmd" exp/tri2b/graph_nosp_tgsmall \
#wsd#       data/$test exp/tri2b/decode_nosp_tgsmall_$test
#wsd#     steps/lmrescore.sh --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tgmed} \
#wsd#       data/$test exp/tri2b/decode_nosp_{tgsmall,tgmed}_$test
#wsd#     steps/lmrescore_const_arpa.sh \
#wsd#       --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tglarge} \
#wsd#       data/$test exp/tri2b/decode_nosp_{tgsmall,tglarge}_$test
#wsd#   done
#wsd# )&

# Align a 10k utts subset using the tri2b model
#wsd# steps/align_si.sh  --nj 10 --cmd "$train_cmd" --use-graphs true \
#wsd#   data/train_10k data/lang_nosp exp/tri2b exp/tri2b_ali_10k

# Train tri3b, which is LDA+MLLT+SAT on 10k utts
#wsd# steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
#wsd#   data/train_10k data/lang_nosp exp/tri2b_ali_10k exp/tri3b

# decode using the tri3b model
#wsd# (
#wsd#   utils/mkgraph.sh data/lang_nosp_test_tgsmall \
#wsd#     exp/tri3b exp/tri3b/graph_nosp_tgsmall
#wsd#   for test in test_clean test_other dev_clean dev_other; do
#wsd#     steps/decode_fmllr.sh --nj 20 --cmd "$decode_cmd" \
#wsd#       exp/tri3b/graph_nosp_tgsmall data/$test \
#wsd#       exp/tri3b/decode_nosp_tgsmall_$test
#wsd#     steps/lmrescore.sh --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tgmed} \
#wsd#       data/$test exp/tri3b/decode_nosp_{tgsmall,tgmed}_$test
#wsd#     steps/lmrescore_const_arpa.sh \
#wsd#       --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tglarge} \
#wsd#       data/$test exp/tri3b/decode_nosp_{tgsmall,tglarge}_$test
#wsd#   done
#wsd# )&

# align the entire train_clean_100 subset using the tri3b model
#wsd# steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" \
#wsd#   data/train_clean_100 data/lang_nosp \
#wsd#   exp/tri3b exp/tri3b_ali_clean_100

# train another LDA+MLLT+SAT system on the entire 100 hour subset
#wsd# steps/train_sat.sh  --cmd "$train_cmd" 4200 40000 \
#wsd#   data/train_clean_100 data/lang_nosp \
#wsd#   exp/tri3b_ali_clean_100 exp/tri4b

# decode using the tri4b model
#wsd# (
#wsd#   utils/mkgraph.sh data/lang_nosp_test_tgsmall \
#wsd#     exp/tri4b exp/tri4b/graph_nosp_tgsmall
#wsd#   for test in test_clean test_other dev_clean dev_other; do
#wsd#     steps/decode_fmllr.sh --nj 20 --cmd "$decode_cmd" \
#wsd#       exp/tri4b/graph_nosp_tgsmall data/$test \
#wsd#       exp/tri4b/decode_nosp_tgsmall_$test
#wsd#     steps/lmrescore.sh --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tgmed} \
#wsd#       data/$test exp/tri4b/decode_nosp_{tgsmall,tgmed}_$test
#wsd#     steps/lmrescore_const_arpa.sh \
#wsd#       --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tglarge} \
#wsd#       data/$test exp/tri4b/decode_nosp_{tgsmall,tglarge}_$test
#wsd#     steps/lmrescore_const_arpa.sh \
#wsd#       --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,fglarge} \
#wsd#       data/$test exp/tri4b/decode_nosp_{tgsmall,fglarge}_$test
#wsd#   done
#wsd# )&

# Now we compute the pronunciation and silence probabilities from training data,
# and re-create the lang directory.
#wsd# steps/get_prons.sh --cmd "$train_cmd" \
#wsd#   data/train_clean_100 data/lang_nosp exp/tri4b
#wsd# utils/dict_dir_add_pronprobs.sh --max-normalize true \
#wsd#   data/local/dict_nosp \
#wsd#   exp/tri4b/pron_counts_nowb.txt exp/tri4b/sil_counts_nowb.txt \
#wsd#   exp/tri4b/pron_bigram_counts_nowb.txt data/local/dict

#wsd# utils/prepare_lang.sh data/local/dict \
#wsd#   "<UNK>" data/local/lang_tmp data/lang
#wsd# local/format_lms.sh --src-dir data/lang data/local/lm

#wsd# utils/build_const_arpa_lm.sh \
#wsd#   data/local/lm/lm_tglarge.arpa.gz data/lang data/lang_test_tglarge
#wsd# utils/build_const_arpa_lm.sh \
#wsd#   data/local/lm/lm_fglarge.arpa.gz data/lang data/lang_test_fglarge

# decode using the tri4b model with pronunciation and silence probabilities
#wsd# (
#wsd#   utils/mkgraph.sh \
#wsd#     data/lang_test_tgsmall exp/tri4b exp/tri4b/graph_tgsmall
#wsd#   for test in test_clean test_other dev_clean dev_other; do
#wsd#     steps/decode_fmllr.sh --nj 20 --cmd "$decode_cmd" \
#wsd#       exp/tri4b/graph_tgsmall data/$test \
#wsd#       exp/tri4b/decode_tgsmall_$test
#wsd#     steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
#wsd#       data/$test exp/tri4b/decode_{tgsmall,tgmed}_$test
#wsd#     steps/lmrescore_const_arpa.sh \
#wsd#       --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
#wsd#       data/$test exp/tri4b/decode_{tgsmall,tglarge}_$test
#wsd#     steps/lmrescore_const_arpa.sh \
#wsd#       --cmd "$decode_cmd" data/lang_test_{tgsmall,fglarge} \
#wsd#       data/$test exp/tri4b/decode_{tgsmall,fglarge}_$test
#wsd#   done
#wsd# )&

# align train_clean_100 using the tri4b model
#wsd# steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
#wsd#   data/train_clean_100 data/lang exp/tri4b exp/tri4b_ali_clean_100

# if you want at this point you can train and test NN model(s) on the 100 hour
# subset
#wsd# local/nnet2/run_5a_clean_100.sh

# wsd #align dev data set 
#wsd4# steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
#wsd4#   data/train_10k data/lang exp/tri4b exp/tri4b_ali_10k
#begin training DNN-HMM system

. ./path.sh
#RBM pretraining
#wsd5# dir=exp/tri5b_rbm_pretrain
#wsd5# mkdir -p $dir
#wsd5# $cuda_cmd $dir/_pretrain_dbn.log \
#wsd5#       steps/nnet/pretrain_dbn.sh --nn-depth 4 --hid-dim 1024 --rbm-iter 3 --splice 10 data-fbank/train_clean_100 $dir
#wsd5# #BP 
#wsd5# dir=exp/tri5b_nnet_fbank
#wsd5# ali=exp/tri4b_ali_clean_100
#wsd5# ali_dev=exp/tri4b_ali_10k
#wsd5# feature_transform=exp/tri5b_rbm_pretrain/final.feature_transform
#wsd5# dbn=exp/tri5b_rbm_pretrain/4.dbn
#wsd5# $cuda_cmd $dir/_train_nnet.log \
#wsd5#       steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.001 \
#wsd5#         data-fbank/train_clean_100 data-fbank/train_10k data/lang $ali $ali_dev $dir || exit 1;
#wsd5# 
#wsd5# utils/mkgraph.sh data/lang_nosp_test_tgsmall exp/tri5b_nnet_fbank exp/tri5b_nnet_fbank/graph_nosp_test_tgsmall || exit 1;
#wsd5# for test in test_clean test_other dev_clean dev_other; do
#wsd5#   steps/nnet/decode.sh --nj 8 --acwt 0.10 --config conf/decode_dnn.config \
#wsd5#       exp/tri5b_nnet_fbank/graph_nosp_test_tgsmall data-fbank/$test $dir/decode_nosp_test_tgsmall_$test || exit 1;
#wsd5# done


#Retrain system using new ali,
#this is essential  
#repeat this process for 3 times 
srcdir=exp/tri5b_nnet_fbank
steps/nnet/align.sh --nj 10 \
      data-fbank/train_clean_100 data/lang $srcdir ${srcdir}_ali_clean_100 || exit 1;
steps/nnet/align.sh --nj 10 \
      data-fbank/train_10k data/lang $srcdir ${srcdir}_ali_10k || exit 1;

#no need to do pretraining again
dir=exp/tri6b_nnet_fbank
ali=exp/tri5b_nnet_fbank_ali_clean_100
ali_dev=exp/tri5b_nnet_fbank_ali_10k
feature_transform=exp/tri5b_rbm_pretrain/final.feature_transform
dbn=exp/tri5b_rbm_pretrain/4.dbn
$cuda_cmd $dir/_train_nnet.log \
      steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.001 \
        data-fbank/train_clean_100 data-fbank/train_10k data/lang $ali $ali_dev $dir || exit 1;

utils/mkgraph.sh data/lang_nosp_test_tgsmall exp/tri6b_nnet_fbank exp/tri6b_nnet_fbank/graph_nosp_test_tgsmall || exit 1;

for test in test_clean test_other dev_clean dev_other; do
  steps/nnet/decode.sh --nj 8 --acwt 0.10 --config conf/decode_dnn.config \
      exp/tri6b_nnet_fbank/graph_nosp_test_tgsmall data-fbank/$test $dir/decode_nosp_$test || exit 1;
done
#wsd4# 
#wsd4# srcdir=exp/tri6b_nnet_fbank
#wsd4# steps/nnet/align.sh --nj 10 \
#wsd4#       data-fbank/train_clean_100 data/lang $srcdir ${srcdir}_ali_clean_100 || exit 1;
#wsd4# steps/nnet/align.sh --nj 10 \
#wsd4#       data-fbank/train_10k data/lang $srcdir ${srcdir}_ali_10k || exit 1;
#wsd4# 
#wsd4# . ./path.sh
#wsd4# dir=exp/tri7b_nnet_fbank
#wsd4# ali=exp/tri6b_nnet_fbank_ali_clean_100
#wsd4# ali_dev=exp/tri6b_nnet_fbank_ali_10k
#wsd4# feature_transform=exp/tri5b_rbm_pretrain/final.feature_transform
#wsd4# dbn=exp/tri5b_rbm_pretrain/4.dbn
#wsd4# $cuda_cmd $dir/_train_nnet.log \
#wsd4#       steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.001 \
#wsd4#         data-fbank/train_clean_100 data-fbank/train_10k data/lang $ali $ali_dev $dir || exit 1;
#wsd4# 
#wsd4# utils/mkgraph.sh data/lang_nosp_test_tgsmall exp/tri7b_nnet_fbank exp/tri7b_nnet_fbank/graph_nosp_test_tgsmall || exit 1;
#wsd4# 
#wsd4# for test in test_clean test_other dev_clean dev_other; do
#wsd4#   steps/nnet/decode.sh --nj 8 --acwt 0.10 --config conf/decode_dnn.config \
#wsd4#       exp/triba_nnet_fbank/graph_nosp_test_tgsmall data-fbank/$test $dir/decode_nosp_$test || exit 1;
#wsd4# done
#wsd3# 
#wsd3# srcdir=exp/tri7b_nnet_fbank
#wsd3# steps/nnet/align.sh --nj 10 \
#wsd3#       data-fbank/train_clean_100 data/lang $srcdir ${srcdir}_ali_clean_100 || exit 1;
#wsd3# steps/nnet/align.sh --nj 10 \
#wsd3#       data-fbank/train_10k data/lang $srcdir ${srcdir}_ali_10k || exit 1;



#wait
