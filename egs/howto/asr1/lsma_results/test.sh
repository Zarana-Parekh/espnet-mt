grep groundtruth $1 > gt.log
grep prediction $1 > preds.log
python refactor.py gt.log > $2
python refactor.py preds.log > $3
rm gt.log
rm  preds.log
