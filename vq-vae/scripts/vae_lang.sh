DIM=$1

for SEED in 1
do
	python main.py --use_language --model vae --seed ${SEED} --dataset coco --dataset_dir_name data/ --data-dir data/coco/ --hidden ${DIM} --batch-size 32 --save-name vae_lang_${DIM}_${SEED};
done
