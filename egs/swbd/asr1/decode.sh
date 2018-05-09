#!/bin/bash

#PBS -M spalaska@cs.cmu.edu
#PBS -j oe
#PBS -o log
#PBS -d .
#PBS -V
#PBS -l walltime=48:00:00
#PBS -l nodes=compute-2-261:ppn=1
##PBS -l nodes=compute-2-262:ppn=1
##PBS -l nodes=compute-2-263:ppn=1
##PBS -l nodes=compute-2-264:ppn=1
#PBS -N dec_word

##SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --output=log/slurm-%j.out
#SBATCH --export=ALL
#SBATCH --time=48:00:00


# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# to run:
: <<'END'
expath=

./decode.sh --ctc_weight 0 --beam_size 20 --penalty 0.1 --expdir $expath --target char --datadir data --dumpdir /tmp/spalaska/swbd_data --dump_h False --recog_set eval2000
END
 . ./path.sh
 . ./cmd.sh

 # general configuration
 backend=pytorch
 stage=5        # start from 0 if you need to start from data preparation
 gpu=0
 ngpu=0          # use 0 when using GPU on slurm/grid engine, otherwise -1
 debugmode=1
 dumpdir=
 datadir=

 # feature configuration
 do_delta=false # true when using CNN

 # decoding parameter
 beam_size=20
 penalty=0.1
 maxlenratio=0.0
 minlenratio=0.0
 lm_weight=0.3
 ctc_weight=0
 mtlalpha=0

 verbose=1

 expdir=
 recog_model=acc.best

 target=char
 adaptation=0
 dump_h=False

 nlsyms=data/lang_1char/non_lang_syms.txt
 #recog_set="dev_test held_out_test"
 #recog_set="dev_test"
 #recog_set="held_out_test"
 recog_set=eval2000

 . utils/parse_options.sh || exit 1;
 . ./path.sh
 . ./cmd.sh

 # Set bash to 'debug' mode, it will exit on :
 # -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
 set -e
 set -u
 set -o pipefail

 uname -n
 echo "expdir: ${expdir}"
 dict=${datadir}/lang_1char/train_${target}_units.txt
 echo "dictionary: ${dict}"

 modeldir=${expdir}/results/model.${recog_model}
 modelconfdir=${expdir}/results/model.conf

 echo "modeldir: ${modeldir}"
 echo "modelconfdir: ${modelconfdir}"

 echo "dump_h: ${dump_h}"

 echo "Recog set: ${recog_set}"

if [ ${stage} -le 5 ]; then
     echo "stage 5: Decoding"
     nj=32

     for rtask in ${recog_set}; do
     (
         decode_dir=decode_${rtask}_beam${beam_size}_p${penalty}_len${minlenratio}-${maxlenratio}_nj${nj}

         # feature extraction
		 feats="ark,s,cs:copy-feats scp:$dumpdir/$rtask/delta${do_delta}/feats.JOB.scp ark:- |"

         if ${do_delta}; then
         feats="$feats add-deltas ark:- ark:- |"
         fi

         #### use CPU for decoding
         ngpu=0

         ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
             asr_recog.py \
             --ngpu ${ngpu} \
			 --verbose ${verbose} \
             --backend ${backend} \
             --recog-feat "$feats" \
             --recog-label ${dumpdir}/${rtask}/delta${do_delta}/data_${target}.json \
             --result-label ${expdir}/${decode_dir}/data.JOB.json \
             --model ${modeldir}  \
             --model-conf ${modelconfdir}  \
             --beam-size ${beam_size} \
             --penalty ${penalty} \
             --maxlenratio ${maxlenratio} \
             --minlenratio ${minlenratio} \
             --ctc-weight ${ctc_weight} \
             --adaptation ${adaptation} \
             --dump_h ${dump_h} &
 #            --rnnlm ${lmexpdir}/rnnlm.model.best \
 #            --lm-weight ${lm_weight} &
         wait

         if [ "${target}" == "bpe" ]; then
             score_sclite.sh --bpe true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}
         else
             score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}
         fi

     ) &
     done
     wait
     echo "Finished"
 fi
