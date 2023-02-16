seeds=(0 27 42)
representationDims=(8x8 4x4)
captioning=(no-captioning_loss captioning_loss)

for i in ${seeds[@]}
do
    for j in ${representationDims[@]}
    do
        echo -e "Current Test: \n"
        echo -e "torchrun --standalone --nnodes=1 --nproc_per_node=10 main_stage1.py -m=configs/coco/stage1/coco-rqvae-${j}x4.yaml -r=RQVAE_results --no-captioning_loss --seed=${i} --CLIPLoss\n\n\n\n"
        torchrun --standalone --nnodes=1 --nproc_per_node=10 main_stage1.py -m=configs/coco/stage1/coco-rqvae-${j}x4.yaml -r=RQVAE_results --no-captioning_loss --seed=${i} --CLIPLoss

    done

done
