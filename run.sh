#!/bin/bash


####################
#  ENGLISH
####################

# Runs the experiment for the english dataset

# Path variables
EVAL1_BIN=/fhgfs/bootphon/scratch/roland/zerospeech2015/english_eval1/eval1
ENGLISH_AUDIO_FOLDER=/fhgfs/bootphon/scratch/roland/zerospeech2015/english_wavs
ENGLISH_VAD_FILE=resources/english_vad


# Internal variables
ENGLISH_FOLDER="english_experiment/"

####################
# PREPARATION

mkdir -p $ENGLISH_FOLDER
# Compute filterbanks
echo "Computing filterbanks"
python abnet/utils/do_fbanks.py ${ENGLISH_FOLDER}"fbanks" ${ENGLISH_AUDIO_FOLDER}/*
# Mean variance normalize fbanks
echo "Normalizing filterbanks"
python zerospeech/renormalize.py ${ENGLISH_FOLDER}"fbanks_normalized" ${ENGLISH_VAD_FILE} ${ENGLISH_FOLDER}fbanks/*
# Preparing training examples
python zerospeech/from_aren.py "resources/english.classes" ${ENGLISH_FOLDER}"abnet.joblib" ${ENGLISH_FOLDER}"fbanks_normalized/" "english"

####################
# TRAINING

# Train abnet
echo "Instanciating and training ABnet"
python abnet/train.py --network-type=AB --dataset-path=${ENGLISH_FOLDER}"abnet.joblib" --dataset-name=${ENGLISH_FOLDER}"english" --nframes=7 --output-file-name=${ENGLISH_FOLDER}"abnet"

####################
# TESTING

# Stack fbanks
echo "Stacking fbanks for evaluation"
python abnet/utils/stack_fbanks.py ${ENGLISH_FOLDER}"fbanks_normalized_stacked" ${ENGLISH_FOLDER}fbanks_normalized/*
# Test abnet
echo "Testing ABnet"
python zerospeech/evaluate.py ${ENGLISH_FOLDER}"abnet.pickle" ${ENGLISH_FOLDER}"fbanks_normalized_stacked" ${ENGLISH_FOLDER}"english_mean_std.npz" ${ENGLISH_FOLDER}"embeddings/" "txt"

# Evaluate embeddings
echo "Evaluating embeddings"
${EVAL1_BIN} -j 10 ${ENGLISH_FOLDER}"embeddings/" ${ENGLISH_FOLDER}"results/"

