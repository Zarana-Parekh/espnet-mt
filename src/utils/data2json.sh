#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

nlsyms=""
lang=""
feat="" # feat.scp
oov="<unk>"
bpecode=""
word_model=false
bpe_model=false
vis_feat=false
obj_feat_path=""
plc_feat_path=""
topic_feat_path=""

. utils/parse_options.sh

if [ $# != 2 ]; then
    echo "Usage: $0 <data-dir> <dict>";
    exit 1;
fi

dir=$1
dic=$2
#tmpdir=`mktemp -d ${dir}/tmp-XXXXX`
tmpdir=`mktemp -d tmp-sp-XXXXX`
#rm -f ${tmpdir}/*.scp

# input, which is not necessary for decoding mode, and make it as an option
if [ ! -z ${feat} ]; then
    feat-to-len scp:${feat} ark,t:${tmpdir}/ilen.scp
    feat-to-dim scp:${feat} ark,t:${tmpdir}/idim.scp
fi

# output
if ${bpe_model}; then
    paste -d " " <(awk '{print $1}' ${dir}/text) <(cut -f 2- -d" " ${dir}/text | ../../../tools/subword-nmt/apply_bpe.py -c ${bpecode}) > ${tmpdir}/token.scp
elif ${word_model}; then
    #text2word_token.py -s 1 -l ${nlsyms} ${dir}/text > ${tmpdir}/token.scp
    text2word_token.py -s 1 ${dir}/text > ${tmpdir}/token.scp
elif [ ! -z ${nlsyms} ]; then
    text2token.py -s 1 -n 1 -l ${nlsyms} ${dir}/text > ${tmpdir}/token.scp
else
    text2token.py -s 1 -n 1 ${dir}/text > ${tmpdir}/token.scp
fi
cat ${tmpdir}/token.scp | utils/sym2int.pl --map-oov ${oov} -f 2- ${dic} > ${tmpdir}/tokenid.scp
cat ${tmpdir}/tokenid.scp | awk '{print $1 " " NF-1}' > ${tmpdir}/olen.scp
# +2 comes from CTC blank and EOS
vocsize=`tail -n 1 ${dic} | awk '{print $2}'`
odim=`echo "$vocsize + 2" | bc`
awk -v odim=${odim} '{print $1 " " odim}' ${dir}/text > ${tmpdir}/odim.scp

# others
if [ ! -z ${lang} ]; then
    awk -v lang=${lang} '{print $1 " " lang}' ${dir}/text > ${tmpdir}/lang.scp
fi

# for including visual feats in data.json
if ${vis_feat}; then
    python ../../../src/utils/pkl2json.py ${obj_feat_path} ${tmpdir} obj
    python ../../../src/utils/pkl2json.py ${plc_feat_path} ${tmpdir} plc
    python ../../../src/utils/pkl2json.py ${topic_feat_path} ${tmpdir} topic
fi

#rm -f ${tmpdir}/*.json
for x in ${dir}/text ${dir}/utt2spk ${tmpdir}/*.scp; do
    k=`basename ${x} .scp`
    cat ${x} | scp2json.py --key ${k} > ${tmpdir}/${k}.json
done
mergejson.py ${tmpdir}/*.json
#rm -fr ${tmpdir}
