#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# to run:
##### /data/ASR5/sreyashn/data/how-to-data/ramon_splits/splits
: <<'END'
./run.mt.sh --backend pytorch --etype blstm --mtlalpha 0 --ctc_weight 0 --dumpdir /tmp/zpp/howto_data_480h --datadir data/480h --expdir_main exp --ngpu 1 --elayer 2 --eunits 500 --dlayers 2 --dunits 500 --eproj 500 --opt adam --atype dot --beam-size 5 --epochs 10 --batchsize 24 --target mt --vis_feat false --adaptation 0 --stage 1
END
. ./path.sh
. ./cmd.sh

# general configuration
backend=chainer
stage=0        # start from 0 if you need to start from data preparation
gpu=           # will be deprecated, please use ngpu
ngpu=0         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
datadir=       # directory pointing to data
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot
seed=1

# feature configuration
do_delta=false # true when using CNN

# network archtecture
# encoder related
etype=vggblstmp     # encoder architecture type
elayers=6
eunits=320
eprojs=320
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
# loss related
ctctype=chainer
# decoder related
dlayers=1
dunits=300
# attention related
atype=location
adim=320
awin=5
aheads=4
aconv_chans=10
aconv_filts=100

# hybrid CTC/attention
mtlalpha=0.5

# label smoothing
lsm_type= #unigram
lsm_weight=0.05

# minibatch related
batchsize=30
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

lm_batchsize=2048
lm_epoch=80

# optimization related
opt=adadelta
epochs=15

# rnnlm related
lm_weight=1.0

# decoding parameter
beam_size=20
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.3
recog_model=loss.best # set a model to be used for decoding: 'acc.best' or 'loss.best'

# target unit related
nbpe=300
initchar=false
target=char
bplen=35

# dumping encoder hidden vector weights or attention weights
dump_h=false
dump_attn=false

# visual feat related
vis_feat=false
obj_feat_path=/data/ASR5/spalaska/pytorch-projects/espnet-avsr/egs/howto/asr1/data/visfeats/howto_480h_obj_1frame_100d.p
plc_feat_path=/data/ASR5/spalaska/pytorch-projects/espnet-avsr/egs/howto/asr1/data/visfeats/place_features.p
topic_feat_path=/data/ASR5/spalaska/pytorch-projects/espnet-avsr/egs/howto/asr1/data/visfeats/topic_features.p
adaptation=0

# data
# ---- put path to wav and transcripts here if needed

# exp tag
tag="" # tag for managing experiments.
expdir_main=

. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

# check gpu option usage
if [ ! -z $gpu ]; then
    echo "WARNING: --gpu option will be deprecated."
    echo "WARNING: please use --ngpu option."
    if [ $gpu -eq -1 ]; then
        ngpu=0
    else
        ngpu=1
    fi
fi

# only for CLSP
if [[ $(hostname -f) == *.clsp.jhu.edu ]] ; then
    export CUDA_VISIBLE_DEVICES=$(/usr/local/bin/free-gpu -n $ngpu)
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
train_dev=dev_test
train_test=held_out_test
recog_set="dev_test held_out_test"

# Different target units
echo $target
if [ "${target}" == "char" ]; then
    bpe_model=false
    word_model=false
    mt_model=false
elif [ "${target}" == "bpe" ]; then
    bpe_model=true
    word_model=false
    mt_model=false
elif [ "${target}" == "word" ]; then
    bpe_model=false
    word_model=true
    mt_model=false
elif [ "${target}" == "mt" ]; then
    bpe_model=false
    word_model=false
    mt_model=true
else
    echo "Wrong target unit specified, exiting."
    exit 1;
fi


feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
feat_te_dir=${dumpdir}/${train_test}/delta${do_delta}; mkdir -p ${feat_te_dir}

if [ ${stage} -le 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases

    echo "stage 1: MT data processing"

    # TODO ZPP add data creation here, dictionary creation, set dict below, json creation
    # TODO: dir with all jsons, will also have vis jsons
    # after merge, you have feature
    tmpdir=/tmp/zpp/final_data

    python ../../../src/utils/pkl2json.py ${obj_feat_path} ${tmpdir} obj
    python ../../../src/utils/pkl2json.py ${plc_feat_path} ${tmpdir} plc
    python ../../../src/utils/pkl2json.py ${topic_feat_path} ${tmpdir} topic

    mergejson.py /tmp/zpp/final_data/text/train/*.json ${tmpdir}/obj_feat.json ${tmpdir}/plc_feat.json ${tmpdir}/topic_feat.json > /tmp/zpp/final_data/train_new.json
    mergejson.py /tmp/zpp/final_data/text/val/*.json ${tmpdir}/obj_feat.json ${tmpdir}/plc_feat.json ${tmpdir}/topic_feat.json > /tmp/zpp/final_data/val_new.json
    mergejson.py /tmp/zpp/final_data/text/test/*.json ${tmpdir}/obj_feat.json ${tmpdir}/plc_feat.json ${tmpdir}/topic_feat.json > /tmp/zpp/final_data/test_new.json
    exit 1;

fi

# path to portuguese dictionary
dict=${datadir}/lang_1char/${train_set}_${target}_units.txt

if [ "${target}" == "bpe" ]; then
    targetname=${target}${nbpe}
else
    targetname=${target}
fi

if [ -z ${tag} ]; then
    ### modified for incorporating adaptation
    expdir=${expdir_main}/${train_set}_adapt${adaptation}_${targetname}_${etype}_e${elayers}_eunit${eunits}_proj${eprojs}_d${dlayers}_dunit${dunits}_${atype}_mtlalpha${mtlalpha}_${opt}_bs${batchsize}_opt${opt}
    if [ "${lsm_type}" != "" ]; then
        expdir=${expdir}_lsm${lsm_type}${lsm_weight}
    fi
    if ${do_delta}; then
        expdir=${expdir}_delta
    fi
else
    expdir=${expdir_main}/${train_set}_${tag}
fi
mkdir -p ${expdir}

#########
if [ ${stage} -le 4 ]; then
    echo "stage 4: Network Training"

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --debugmode ${debugmode} \
        --dict /tmp/zpp/final_data/mapping/word_idx_pt.txt \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --seed ${seed} \
        --train-feat  /tmp/zpp/final_data/train_new.json \
        --valid-feat /tmp/zpp/final_data/val_new.json \
        --train-label /tmp/zpp/final_data/train_new.json \
        --valid-label /tmp/zpp/final_data/val_new.json \
        --etype ${etype} \
        --elayers ${elayers} \
        --eunits ${eunits} \
        --eprojs ${eprojs} \
        --subsample ${subsample} \
        --ctc_type ${ctctype} \
        --dlayers ${dlayers} \
        --dunits ${dunits} \
        --atype ${atype} \
        --adim ${adim} \
        --awin ${awin} \
        --aheads ${aheads} \
        --aconv-chans ${aconv_chans} \
        --aconv-filts ${aconv_filts} \
        --mtlalpha ${mtlalpha} \
        --lsm-type ${lsm_type} \
        --lsm-weight ${lsm_weight} \
        --batch-size ${batchsize} \
        --maxlen-in ${maxlen_in} \
        --maxlen-out ${maxlen_out} \
        --opt ${opt} \
        --dropout-rate 0.3 \
        --epochs ${epochs} \
     	--adaptation ${adaptation}
fi

if [ ${stage} -le 5 ]; then
    echo "stage 5: Decoding"
    nj=1

    #for rtask in ${recog_set}; do
    #(
        # modified for incorporating adaptation
        #decode_dir=decode_final_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}
        decode_dir=decode_final_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}
        echo $decode_dir

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-feat /tmp/zpp/final_data/small_test.json \
            --recog-label /tmp/zpp/final_data/small_test.json \
            --result-label /tmp/zpp/final_data/decode/data.JOB.json \
            --model ${expdir}/results/model.${recog_model}  \
            --model-conf ${expdir}/results/model.conf  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --adaptation ${adaptation} \
            --ctc-weight ${ctc_weight}
        wait

        if [ "${target}" == "bpe" ]; then
            #score_sclite.sh --bpe true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}
            score_sclite.sh --bpe true ${expdir}/${decode_dir} ${dict}
        elif [ "${target}" == "mt" ]; then
            echo '----- this is a mt decoding task ----'
        else
            #score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}
            score_sclite.sh --wer true ${expdir}/${decode_dir} ${dict}
        fi

    #) &
    #done
    wait
    echo "Finished"
fi

#            --adaptation ${adaptation} &
#            --rnnlm ${lmexpdir}/rnnlm.model.best \
#            --lm-weight ${lm_weight} &
