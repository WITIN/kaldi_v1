#!/bin/bash


# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
data=/data/kaldi/egs/librispeech/s5/data
data_url=www.openslr.org/resources/12
lm_url=www.openslr.org/resources/11
. ./cmd.sh
. ./path.sh


# download the data.  Note: we're using the 100 hour setup for
# now; later in the script we'll download more and use it to train neural
# nets.
for part in dev-clean test-clean dev-other test-other train-clean-100; do
  local/download_and_untar.sh $data $data_url $part
done

# download the LM resources
local/download_lm.sh $lm_url data/local/lm

local/download_and_untar.sh $data $data_url train-clean-360

local/download_and_untar.sh $data $data_url train-other-500

