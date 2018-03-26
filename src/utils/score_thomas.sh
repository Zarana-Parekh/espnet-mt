#!/bin/bash

#source /home/tzenkel/thesisEnv/bin/activate

#evalStm="/data/ASR5/ramons/projects/pnc/eesen/asr_egs/swbd/v2_pnc/data/eval2000/stm"
evalStm="/data/ASR5/ramons_2/sinbad_projects/youtube_project/am/eesen_20170714/asr_egs/swbd/v1-tf/data/eval2000/stm"
#evalGlm="/data/ASR5/ramons_2/sinbad_projects/youtube_project/am/eesen_20170714/asr_egs/swbd/v1-tf/data/eval2000/glm"
#evalStm="/data/ASR5/spalaska/espnet/egs/swbd/asr1/data/eval2000/stm"

evalGlm="/data/ASR5/fmetze/eesen/tools/sctk/src/md-eval/test/en20030506.glm"

if [ "$#" -eq 1 ]; then
  evalStm=`expr $1`
  echo "using ${evalStm}"
fi

mkdir -p score
#cat *.txt > score/hyp.text

python /data/ASR5/spalaska/espnet/src/utils/trn2txt.py hyp.trn
python /data/ASR5/tzenkel/masterthesis/code/pythonScripts/textToCtm.py -t hyp.txt -o hyp.ctm
cp hyp.ctm score/ 
cd score
cp $evalStm stm
sort +0 -1 +1 -2 +3nb -4 hyp.ctm > hyp.sort.ctm
# create callhome and switchboard files
grep "en_" hyp.sort.ctm > hyp.sort.callhm.ctm 
grep "sw_" hyp.sort.ctm > hyp.sort.swbd.ctm
grep "en_" stm > stm.callhm
grep "sw_" stm > stm.swbd


/data/ASR5/tzenkel/eesen/tools/sctk/bin/hubscr.pl -p /data/ASR5/tzenkel/eesen/tools/sctk/bin/ -V -l english -h hub5 -g $evalGlm -r stm.callhm hyp.sort.callhm.ctm

/data/ASR5/tzenkel/eesen/tools/sctk/bin/hubscr.pl -p /data/ASR5/tzenkel/eesen/tools/sctk/bin/ -V -l english -h hub5 -g $evalGlm -r stm.swbd hyp.sort.swbd.ctm

/data/ASR5/tzenkel/eesen/tools/sctk/bin/hubscr.pl -p /data/ASR5/tzenkel/eesen/tools/sctk/bin/ -V -l english -h hub5 -g $evalGlm -r stm hyp.sort.ctm

#deactivate
