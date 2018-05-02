#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# to run:
: <<'END'
initpath=exp/90h/train_char_blstmp_e6_subsample1_2_2_1_1_unit320_proj320_ctcchainer_d1_unit300_location_aconvc10_aconvf100_mtlalpha0_adadelta_bs44_mli800_mlo150_lsmunigram0.05/results/model.acc.best

./run.dumph.sh --backend pytorch --etype blstmp --mtlalpha 0 --ctc_weight 0 --dumpdir /tmp/spalaska/howto_data_480h --datadir data/480h --expdir_main exp/480h --target word --initchar false --vis_feat false --stage 5 --dump_h true

./run.sh --backend pytorch --etype blstmp --mtlalpha 0 --ctc_weight 0 --dumpdir /tmp/spalaska/howto_data --datadir data/90h --expdir_main exp/90h --ngpu 1 --epochs 20 --batchsize 48 --lm_weight 0.3 --bplen 35 --lm_epoch 50 --target bpe --nbpe 300 --initchar $initpath --vis_feat false --stage 4
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
lsm_type=unigram
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
recog_model=acc.best # set a model to be used for decoding: 'acc.best' or 'loss.best'

# target unit related
nbpe=300
initchar=
target=char
bplen=35

# dumping encoder hidden vector weights or attention weights
dump_h=false
dump_attn=false

# visual feat related
vis_feat=false
adaptation=0
obj_feat_path=/data/ASR5/abhinav5/YTubeV2_480h/object_features.p
plc_feat_path=/data/ASR5/abhinav5/PlacesAlexNet_480h/place_features.p

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
recog_set="dev_test held_out_test"

# Different target units
echo $target
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
    echo "Wrong target unit specified, exiting."
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
    for x in train dev_test dev5_test; do # held_out_test; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 16 ${datadir}/${x} ${expdir_main}/make_fbank/${x} ${fbankdir}
    done

    # compute global CMVN
    compute-cmvn-stats scp:${datadir}/${train_set}/feats.scp ${datadir}/${train_set}/cmvn.ark

    # dump features for training
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
        ${datadir}/${train_set}/feats.scp ${datadir}/${train_set}/cmvn.ark ${expdir_main}/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
        ${datadir}/${train_dev}/feats.scp ${datadir}/${train_set}/cmvn.ark ${expdir_main}/dump_feats/dev ${feat_dt_dir}

    echo "cleaning transcripts"
    ../../../src/utils/clean_transcripts.py ${datadir}/${train_set}/text
    ../../../src/utils/clean_transcripts.py ${datadir}/${train_dev}/text
    ../../../src/utils/clean_transcripts.py ${datadir}/held_out_test/text

fi

dict=${datadir}/lang_1char/${train_set}_${target}_units.txt
nlsyms=${datadir}/lang_1char/non_lang_syms.txt
code=${datadir}/lang_1char/bpe_code${nbpe}.txt

echo "dictionary: ${dict}"
if [ ${stage} -le 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p ${datadir}/lang_1char/

    echo "make a dictionary"

    if [ "${target}" == "char" ]; then
        echo "Character model"
        echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
        # keeping only those units that occur more than 50 times
        text2token.py -s 1 -n 1 ${datadir}/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
            | sort | uniq -c | awk '$1>=50{print $2}' | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    elif [ "${target}" == "bpe" ]; then
        echo "BPE model"
        echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
        echo "<space> 2" >> ${dict}
        # learn bpe units, keeping only those units that occur more than 50 times
        cut -f 2- -d" " ${datadir}/${train_set}/text | ../../../tools/subword-nmt/learn_bpe.py -s  ${nbpe} > ${code}
        cut -f 2- -d" " ${datadir}/${train_set}/text | ../../../tools/subword-nmt/apply_bpe.py -c  ${code} \
            | tr ' ' '\n' | sort | uniq -c | awk '$1>=50{print $2}' | awk '{print $0 " " NR+2}' >> ${dict}
    elif [ "${target}" == "word" ]; then
        echo "Word model"
        #cut -f 2- -d" " ${datadir}/${train_set}/text | tr " " "\n" | sort | uniq -c | awk '$1>=5{print $2}'\
        #    | awk '{print $0 " " NR+1}' >> ${dict}
        ../../../src/utils/make_word_dict.py ${datadir}/${train_set}/text ${dict}
    else
        echo "Wrong target units specified, exiting."
        exit 1;
    fi
    wc -l ${dict}

    echo "make json files"
    #data2json.sh --feat ${feat_tr_dir}/feats.scp --nlsyms ${nlsyms} \
    data2json.sh --feat ${feat_tr_dir}/feats.scp \
        --word_model ${word_model} --bpe_model ${bpe_model} --bpecode ${code} --vis_feat ${vis_feat} \
        --obj_feat_path ${obj_feat_path} --plc_feat_path ${plc_feat_path} \
         ${datadir}/${train_set} ${dict} > ${feat_tr_dir}/data_${target}.json
    #data2json.sh --feat ${feat_dt_dir}/feats.scp --nlsyms ${nlsyms} \
    data2json.sh --feat ${feat_dt_dir}/feats.scp \
        --word_model ${word_model} --bpe_model ${bpe_model} --bpecode ${code} --vis_feat ${vis_feat} \
        --obj_feat_path ${obj_feat_path} --plc_feat_path ${plc_feat_path} \
         ${datadir}/${train_dev} ${dict} > ${feat_dt_dir}/data_${target}.json
fi

# It takes a few days. If you just want to end-to-end ASR without LM,
# you can skip this and remove --rnnlm option in the recognition (stage 5)
lmexpdir=${expdir_main}/train_${target}_rnnlm_2layer_bs2048
mkdir -p ${lmexpdir}
if [ ${stage} -le -999 ]; then
    echo "stage 3: LM Preparation"
    lmdatadir=${datadir}/local/lm_train_${target}
    mkdir -p ${lmdatadir}
    if [ "${target}" == "char" ]; then
        echo "Character model"
        #text2token.py -s 1 -n 1 -l ${nlsyms} ${datadir}/${train_set}/text | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' \
        text2token.py -s 1 -n 1 ${datadir}/${train_set}/text | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' \
        > ${lmdatadir}/train_trans.txt
        cat ${lmdatadir}/train_trans.txt | tr '\n' ' ' > ${lmdatadir}/train.txt
        #text2token.py -s 1 -n 1 -l ${nlsyms} ${datadir}/${train_dev}/text | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' \
        text2token.py -s 1 -n 1 ${datadir}/${train_dev}/text | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' \
        > ${lmdatadir}/valid.txt
    elif [ "${target}" == "bpe" ]; then
        echo "BPE model"
        cut -f 2- -d" " ${datadir}/${train_set}/text | ../../../tools/subword-nmt/apply_bpe.py -c ${code} | perl -pe 's/\n/ <eos> /g' > ${lmdatadir}/train_trans.txt
        cat ${lmdatadir}/train_trans.txt | tr '\n' ' ' > ${lmdatadir}/train.txt
        cut -f 2- -d" " ${datadir}/${train_dev}/text | ../../../tools/subword-nmt/apply_bpe.py -c ${code} | perl -pe 's/\n/ <eos> /g' > ${lmdatadir}/valid.txt
    elif [ "${target}" == "word" ]; then
        echo "Word model"
        cut -f 2- -d" " ${datadir}/${train_set}/text | perl -pe 's/\n/ <eos> /g' \
                         > ${lmdatadir}/train_trans.txt
        cat ${lmdatadir}/train_trans.txt | tr '\n' ' ' > ${lmdatadir}/train.txt
        cut -f 2- -d" " ${datadir}/${train_dev}/text | perl -pe 's/\n/ <eos> /g' \
                         > ${lmdatadir}/valid.txt
    else
        echo "Wrong target units specified, exiting."
        exit 1;
    fi
    # use only 1 gpu
    if [ ${ngpu} -gt 1 ]; then
        echo "LM training does not support multi-gpu. signle gpu will be used."
        lmngpu=1
    else
        lmngpu=0
    fi
    ${cuda_cmd} ${lmexpdir}/train.log \
        lm_train.py \
        --ngpu ${lmngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --batchsize ${lm_batchsize} \
        --epoch ${lm_epoch} \
        --dict ${dict} \
        --bproplen ${bplen}

    echo "LM training finished"
fi

if [ "${target}" == "bpe" ]; then
    targetname=${target}${nbpe}
else
    targetname=${target}
fi

if [ -z ${tag} ]; then
    expdir=${expdir_main}/word_model_dump_h
else
    expdir=${expdir_main}/${train_set}_${tag}
fi
mkdir -p ${expdir}

if [ ${stage} -le 4 ]; then
    echo "stage 4: Network Training"

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --seed ${seed} \
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
        --epochs ${epochs} \
        --initchar ${initchar}
fi

if [ ${stage} -le 5 ]; then
    echo "stage 5: Decoding"
    nj=32
    beam_size=1
    cp exp/480h/train_word_blstmp_e6_subsample1_2_2_1_1_unit320_proj320_ctcchainer_d1_unit300_location_aconvc10_aconvf100_mtlalpha0_adadelta_bs48_mli800_mlo150_lsmunigram0.05/results/model.conf ${expdir}
    cp exp/480h/train_word_blstmp_e6_subsample1_2_2_1_1_unit320_proj320_ctcchainer_d1_unit300_location_aconvc10_aconvf100_mtlalpha0_adadelta_bs48_mli800_mlo150_lsmunigram0.05/results/model.acc.best ${expdir}
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=25:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-feat ${feat_tr_dir}/feats.JOB.ark\
            --save_ark ${feat_tr_dir}/enc_embed.JOB.ark\
            --recog-label ${feat_tr_dir}/data_${target}.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/model.${recog_model}  \
            --model-conf ${expdir}/model.conf  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} \
            --adaptation ${adaptation} \
            --dump_h ${dump_h} &
        wait

    ) &
    done
    wait
    echo "Finished"
fi

