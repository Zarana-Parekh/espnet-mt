#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# to run:
# ./run.sh --backend pytorch --etype blstmp --mtlalpha 0 --ctc_weight 0 --dumpdir /tmp/spalaska/fisher_data --datadir data --gpu 0 --epochs 25 --target char --batchsize 48 --stage 1

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=0        # start from 0 if you need to start from data preparation
gpu=-1         # use 0 when using GPU on slurm/grid engine, otherwise -1
debugmode=1
dumpdir=       # directory to dump full features
datadir=       # directory pointing to data
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot
seed=1

# feature configuration
do_delta=false # true when using CNN

# network archtecture
# encoder related
etype=blstmp # encoder architecture type
elayers=8
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
aconv_chans=10
aconv_filts=100

# hybrid CTC/attention
mtlalpha=0.5

# label smoothing
lsm_type=unigram
lsm_weight=0.05

# minibatch related
batchsize=60
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

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
recog_model=acc.best # set a model to be used for decoding: 'acc.best' or 'loss.best'

# target unit related
nbpe=500
initchar=false
target=

# dumping weights or attention
dump_h=false
dump_att=false

# data
fisher_dir="/data/MM1/corpora/LDC2004T19 /data/MM1/corpora/LDC2005T19 /data/MM1/corpora/LDC2004S13 /data/MM1/corpora/LDC2005S13"
swbd1_dir="/data/MM1/corpora/LDC97S62"
eval2000_dir="/data/MM1/corpora/LDC2002S09/hub5e_00 /data/MM1/corpora/LDC2002T43"
rt03_dir=/data/MM1/corpora/LDC2007S10

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# this train folder contains fisher+swbd data without repetitions
train_set=train_all_nodup
train_dev="train_dev"
recog_set=eval2000 # rt03"

# This code is not consistent for this data
# We only process eval2000 using this, but copy train_all_nodup from ramons folder
# Do not use this stage=0
if [ ${stage} -le 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"

    # training data
    local/fisher_data_prep.sh ${fisher_dir}
    local/swbd1_data_download.sh ${swbd1_dir}
    chmod 644 data/local/dict_nosp/lexicon0.txt
    local/fisher_swbd_prepare_dict.sh
    local/swbd1_data_prep.sh ${swbd1_dir}
    utils/combine_data.sh data/train_all data/train_fisher data/train_swbd

    # test data
    local/eval2000_data_prep.sh ${eval2000_dir}
    local/rt03_data_prep.sh ${rt03_dir}
    # upsample audio from 8k to 16k to make a recipe consistent with others
    for x in train_all eval2000 rt03; do
	sed -i.bak -e "s/$/ sox -R -t wav - -t wav - rate 16000 dither | /" data/${x}/wav.scp
    done
    # normalize eval2000 ant rt03 texts by
    # 1) convert upper to lower
    # 2) remove tags (%AH) (%HESITATION) (%UH)
    # 3) remove <B_ASIDE> <E_ASIDE>
    # 4) remove "(" or ")"
    for x in ${recog_set}; do
        cp data/${x}/text data/${x}/text.org
        paste -d "" \
            <(cut -f 1 -d" " data/${x}/text.org) \
            <(awk '{$1=""; print tolower($0)}' data/${x}/text.org | perl -pe 's| \(\%.*\)||g' | perl -pe 's| \<.*\>||g' | sed -e "s/(//g" -e "s/)//g") \
            | sed -e 's/\s\+/ /g' > data/${x}/text
        # rm data/${x}/text.org
    done
fi

# Different target units
if [ "${target}" == "char" ]; then
    bpe_model=false
    word_model=false
elif [ "${target}" == "bpe" ]; then
    bpe_model=true
    word_model=false
elif [ "${target}" == "word" ]; then
    bpe_model=false
    word_model=true
else
    echo "Wrong target units, exiting."
    exit 1;
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in train_all_nodup ${recog_set}; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 data/${x} exp/make_fbank/${x} ${fbankdir}
    done

    # remove utt having more than 3000 frames or less than 10 frames or
    # remove utt having more than 400 characters or no more than 0 characters
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/train_all_nodup data/${train_set}
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/train_dev data/${train_dev}

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 10 --do_delta $do_delta \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
fi

dict=data/lang_1char/${train_set}_${target}_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt
if [ "${target}" == "bpe" ]; then
    code=data/lang_1char/bpe_code${nbpe}.txt
fi

echo "dictionary: ${dict}"
if [ ${stage} -le 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list"
    cut -f 2- data/${train_set}/text | tr " " "\n" | sort | uniq | grep "\[" > ${nlsyms}
    cat ${nlsyms}

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    if [ "${target}" == "char" ]; then
        echo "Character model"
        text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
        | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    elif [ "${target}" == "bpe" ]; then
        echo "BPE model"
        echo "<space> 2" >> ${dict}
        # learn bpe units
        cut -f 2- -d" " data/${train_set}/text | ../../../tools/subword-nmt/learn_bpe.py -s ${nbpe} > ${code}
        cut -f 2- -d" " data/${train_set}/text | ../../../tools/subword-nmt/apply_bpe.py -c ${code} \
            | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+2}' >> ${dict}
    elif [ "${target}" == "word" ]; then
        echo "Word model"
        cut -f 2- -d" " data/${train_set}/text | tr " " "\n" | sort | uniq -c | awk '$1>=5{print $2}'\
        | awk '{print $0 " " NR+1}' >> ${dict}
    else
        echo "Wrong target unit specified, exiting."
        exit 1;
    fi
    wc -l ${dict}

    echo "make json files"
    data2json.sh --feat ${feat_tr_dir}/feats.scp --nlsyms ${nlsyms} \
        --word_model ${word_model} --bpe_model ${bpe_model} --bpecode ${code} \
         data/${train_set} ${dict} > ${feat_tr_dir}/data_${target}.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --nlsyms ${nlsyms} \
        --word_model ${word_model} --bpe_model ${bpe_model} --bpecode ${code} \
         data/${train_dev} ${dict} > ${feat_dt_dir}/data_${target}.json
fi

if [ -z ${tag} ]; then
    expdir=exp/${train_set}_${target}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_ctc${ctctype}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}
    if ${do_delta}; then
        expdir=${expdir}_delta
    fi
else
    expdir=exp/${train_set}_${tag}
fi
mkdir -p ${expdir}

lmexpdir=exp/train_${target}_rnnlm_2layer_bs256
mkdir -p ${lmexpdir}
if [ ${stage} -le 3 ]; then
    (
    echo "stage 3: LM Preparation"
    lmdatadir=data/local/lm_train_${target}
    mkdir -p ${lmdatadir}
    if [ "${target}" == "char" ]; then
        echo "Character model"
        text2token.py -s 1 -n 1 -l ${nlsyms} data/train_all_nodup/text | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' > ${lmdatadir}/train.txt
        text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_dev}/text | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' > ${lmdatadir}/valid.txt
    elif [ "${target}" == "bpe" ]; then
        echo "BPE model"
        cut -f 2- -d" " data/${train_set}/text | ../../../tools/subword-nmt/apply_bpe.py -c ${code} | perl -pe 's/\n/ <eos> /g' \
            > ${lmdatadir}/train_trans.txt
        cat ${lmdatadir}/train_trans.txt | tr '\n' ' ' > ${lmdatadir}/train.txt
        cut -f 2- -d" " data/${train_dev}/text | ../../../tools/subword-nmt/apply_bpe.py -c ${code} | perl -pe 's/\n/ <eos> /g' \
            > ${lmdatadir}/valid.txt
    elif [ "${target}" == "word" ]; then
        echo "Word model"
        cut -f 2- -d" " data/${train_set}/text | perl -pe 's/\n/ <eos> /g' \
            > ${lmdatadir}/train_trans.txt
        cat ${lmdatadir}/train_trans.txt | tr '\n' ' ' > ${lmdatadir}/train.txt
        cut -f 2- -d" " data/${train_dev}/text | perl -pe 's/\n/ <eos> /g' \
            > ${lmdatadir}/valid.txt
    else
        echo "Wrong target unit specified, exiting"
        exit 1;
    fi
    ${cuda_cmd} ${lmexpdir}/train.log \
        lm_train.py \
        --gpu ${gpu} \
        --backend ${backend} \
        --verbose 1 \
        --batchsize 256 \
        --outdir ${lmexpdir} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --dict ${dict}
    ) &
fi

if [ ${stage} -le 4 ]; then
    echo "stage 4: Network Training"

    ${cuda_cmd} ${expdir}/train.log \
        asr_train.py \
        --gpu ${gpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-feat scp:${feat_tr_dir}/feats.scp \
        --valid-feat scp:${feat_dt_dir}/feats.scp \
        --train-label ${feat_tr_dir}/data_${target}.json \
        --valid-label ${feat_dt_dir}/data_${target}.json \
        --etype ${etype} \
        --elayers ${elayers} \
        --eunits ${eunits} \
        --eprojs ${eprojs} \
        --subsample ${subsample} \
        --ctc_type ${ctctype} \
        --dlayers ${dlayers} \
        --dunits ${dunits} \
        --atype ${atype} \
        --aconv-chans ${aconv_chans} \
        --aconv-filts ${aconv_filts} \
        --mtlalpha ${mtlalpha} \
        --batch-size ${batchsize} \
        --maxlen-in ${maxlen_in} \
        --maxlen-out ${maxlen_out} \
        --opt ${opt} \
        --epochs ${epochs}
fi

wait # wait LM train to be finished
if [ ${stage} -le 5 ]; then
    echo "stage 5: Decoding"
    nj=32

    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}

        # split data
        data=data/${rtask}
        split_data.sh --per-utt ${data} ${nj};
        sdata=${data}/split${nj}utt;

        # feature extraction
        feats="ark,s,cs:apply-cmvn --norm-vars=true data/${train_set}/cmvn.ark scp:${sdata}/JOB/feats.scp ark:- |"
        if ${do_delta}; then
        feats="$feats add-deltas ark:- ark:- |"
        fi

        # make json labels for recognition
        data2json.sh --word_model ${word_model} --bpe_model ${bpe_model} --bpecode ${code} --nlsyms ${nlsyms} ${data} ${dict} > ${data}/data_${target}.json

        #### use CPU for decoding
        gpu=-1

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --gpu ${gpu} \
            --backend ${backend} \
            --recog-feat "$feats" \
            --recog-label ${data}/data_${target}.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/model.${recog_model}  \
            --model-conf ${expdir}/results/model.conf  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} \
            --rnnlm ${lmexpdir}/rnnlm.model.best \
            --lm-weight ${lm_weight} \
            --dump_h ${dump_h} \
            --dump_attn ${dump_attn} &
        wait

        if [ "$target" == "bpe" ]; then
            score_sclite.sh --bpe true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}
        else
            score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}
        fi

    ) &
    done
    wait
    echo "Finished"
fi

