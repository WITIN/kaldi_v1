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
#wsd# 
#wsd# # download the LM resources
#wsd# local/download_lm.sh $lm_url data/local/lm
#wsd# 
#wsd# # format the data as Kaldi data directories
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

mfccdir=mfcc
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

#wsd2# fbankdir=fbank
#wsd2# #wsd2# mkdir -p data-fbank
#wsd2# for part in dev_clean test_clean dev_other test_other train_10k train_clean_100; do
#wsd2# #wsd2#   cp -r data/$part data-fbank/$part
#wsd2# #wsd2#   steps/make_fbank.sh --cmd "$train_cmd" --nj 40 data-fbank/$part exp/make_fbank/$part $fbankdir
#wsd2#   steps/compute_cmvn_stats.sh data-fbank/$part exp/make_fbank/$part $fbankdir
#wsd2# done
#### <wsd fbank>> ############

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
#)&

# align train_clean_100 using the tri4b model
#wsd# steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
#wsd#   data/train_clean_100 data/lang exp/tri4b exp/tri4b_ali_clean_100

# if you want at this point you can train and test NN model(s) on the 100 hour
# subset
#wsd# local/nnet2/run_5a_clean_100.sh

#wsd# #align dev data set 
#wsd# steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
#wsd#  data/train_10k data/lang exp/tri4b exp/tri4b_ali_10k

######## tanh stars below
# DNN hybrid system training parameters


if $use_gpu; then
  if ! cuda-compiled; then
    cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
  fi  
  parallel_opts="--gpu 1"
  num_threads=1
  minibatch_size=512
  dir=exp/tri5c_nnet2_tanh_5x1024
else
  # with just 4 jobs this might be a little slow.
  num_threads=16
  parallel_opts="--num-threads $num_threads"
  minibatch_size=128
  dir=exp/tri5c_nnet2_tanh
fi
. utils/parse_options.sh

train_stage=-10
dnn_mem_reqs="--mem 8G"
#dnn_train_extra_opts="--num-epochs 20 --num-epochs-extra 5 --add-layers-period 1 --shrink-interval 3"
if [ ! -f $dir/final.mdl ]; then
#      --parallel-opts "$parallel_opts" \
#      "${dnn_train_extra_opts[@]}" \ 
#      --splice-width 6 \
  steps/nnet2/train_tanh.sh \
      --stage $train_stage \
      --num-threads "$num_threads" \
      --minibatch-size "$minibatch_size" \
      --initial-learning-rate 0.01 \
      --num-jobs-nnet 4 \
      --final-learning-rate 0.001 \
      --num-hidden-layers 5  \
      --hidden-layer-dim 1024 \
      --left-context 15 \
      --right-context 5 \
      --samples-per-iter 400000 \
      --cmd "$train_cmd" \
      --feat-type raw \
      --num-epochs 20 --num-epochs-extra 5 --add-layers-period 1 --shrink-interval 3 \
  data-fbank/train_clean_100 data/lang exp/tri4b_ali_clean_100 $dir || exit 1
fi
  
for test in test_clean test_other dev_clean dev_other; do
#wsd#    --transform-dir exp/tri4b/decode_tgsmall_$test \
  steps/nnet2/decode.sh --nj 30 --cmd "$decode_cmd" --feat-type raw \
    exp/tri4b/graph_tgsmall data-fbank/$test $dir/decode_tgsmall_$test || exit 1;
  steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
    data-fbank/$test $dir/decode_{tgsmall,tgmed}_$test  || exit 1;
  steps/lmrescore_const_arpa.sh \
    --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
    data-fbank/$test $dir/decode_{tgsmall,tglarge}_$test || exit 1;
  steps/lmrescore_const_arpa.sh \
    --cmd "$decode_cmd" data/lang_test_{tgsmall,fglarge} \
    data-fbank/$test $dir/decode_{tgsmall,fglarge}_$test || exit 1;
done


#  [ ! -d exp/tri4_nnet/decode_dev ] && mkdir -p exp/tri4_nnet/decode_dev
#  decode_extra_opts=(--num-threads 6) 
#  steps/nnet2/decode.sh --cmd "$decode_cmd" --nj "$decode_nj" "${decode_extra_opts[@]}" \
#        --transform-dir exp/tri3/decode_dev exp/tri3/graph data/dev \
#	  exp/tri4_nnet/decode_dev | tee exp/tri4_nnet/decode_dev/decode.log
  
#  [ ! -d exp/tri4_nnet/decode_test ] && mkdir -p exp/tri4_nnet/decode_test
#  steps/nnet2/decode.sh --cmd "$decode_cmd" --nj "$decode_nj" "${decode_extra_opts[@]}" \
#        --transform-dir exp/tri3/decode_test exp/tri3/graph data/test \
#	  exp/tri4_nnet/decode_test | tee exp/tri4_nnet/decode_test/decode.log
  
wait
