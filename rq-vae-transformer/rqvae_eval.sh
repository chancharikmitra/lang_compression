tests=(04012023_034720  04012023_084657  05012023_073037  05012023_122839  06012023_041055  06012023_090853 04012023_170926  05012023_020353  05012023_172623  05012023_224751  06012023_140800  06012023_193158)
dims=(8x8 4x4)
for i in ${dims[@]}
    do
    for j in ${tests[@]}
        do
            python compute_rfid.py --split=val --vqvae=RQVAE_results/coco-rqvae-${i}x4/${j}/epoch10_model.pt
        done
    done
