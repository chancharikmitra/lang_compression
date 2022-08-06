DIM=$1

for SEED in 1
do
	python main.py --model vqvae --use_language --dataset coco --seed ${SEED} --dataset_dir_name data/ --data-dir data/coco/ --k ${DIM} --hidden 64 --batch-size 32 --save-name vqvae_lang_${DIM}_${SEED}
	
done
