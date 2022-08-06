DIM=$1

for SEED in 1
do
	python main.py --model ae --seed ${SEED} --dataset coco --dataset_dir_name data/ --data-dir data/coco/ --hidden ${DIM} --batch-size 32 --save-name ae_${DIM}_${SEED};
done
