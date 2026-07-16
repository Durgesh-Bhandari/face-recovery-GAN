# Face Recovery GAN

Unsupervised joint face rotation + de-occlusion from single image. Two-stage pipeline: 3DDFA frontalization → GAN inpainting.

## Run in Google Colab (T4 GPU)

### 1. Open Notebook

Upload `face_recovery_gan.ipynb` to [Google Colab](https://colab.research.google.com) and select **Runtime → Change runtime type → T4 GPU**.

Or upload this whole repo to Google Drive and open the `.ipynb` from Drive.

### 2. Install Dependencies

```python
!pip install torch torchvision opencv-python numpy tqdm pyyaml tensorboard \
  scikit-image facenet-pytorch scipy albumentations gdown
```

### 3. Download Dataset

```python
!python download_celebA.py --root ./data/celebA --subset 20000
!python download_masks.py --root ./data/masks
```

CelebA is ~1.5GB for 20K images. The mask dataset is small.

### 4. Train

```python
!python train.py --config configs/config.yaml
```

**Training time:** ~3-5 hours on T4 for 80 epochs at 256×256, batch 16.

**Monitor:** TensorBoard runs automatically in the notebook (`%tensorboard --logdir runs`).

### 5. Export Checkpoint

After training, download the model:

```python
from google.colab import files
files.download('./checkpoints/final.pt')
```

### 6. Run Inference (locally, after training)

```bash
python test.py --config configs/config.yaml \
  --checkpoint checkpoints/final.pt \
  --input test_image.jpg \
  --output result.jpg \
  --occlude rect
```

---

## Config Quick Reference

| Param              | Config Key                | Default                  | Notes                       |
| ------------------ | ------------------------- | ------------------------ | --------------------------- |
| Image size         | `data.img_size`           | 256                      | Lower to 128 if VRAM issues |
| Batch size         | `training.batch_size`     | 16                       | Reduce to 8 on smaller GPUs |
| Epochs             | `training.epochs`         | 80                       | 30 enough for PoC           |
| Occlusion types    | `augmentation.mask_types` | irregular, rect, half    | Add/remove as needed        |
| Generator channels | `generator.channels`      | [64,128,256,512,512,512] | Reduce depth for speed      |

## File Structure

```
├── configs/config.yaml        # All training hyperparameters
├── data/
│   ├── mask_generator.py      # Synthetic occlusion masks
│   └── dataloader.py          # Dataset + DataLoader
├── models/
│   ├── generator.py           # Contextual UNet (28M params)
│   ├── discriminator.py       # SN-PatchGAN (2.7M params)
│   ├── losses.py              # L1 + perceptual + style + GAN + ID loss
│   └── face_model.py          # 3DDFA frontalization wrapper
├── utils/
│   ├── metrics.py             # PSNR, SSIM, FID
│   └── visualize.py           # TensorBoard helpers
├── train.py                   # Main training loop
├── test.py                    # Inference script
└── face_recovery_gan.ipynb    # Colab notebook
```

## Notes

- **3DDFA frontalization**: Optional. Defaults to affine fallback if 3DDFA not installed. Install separately for full pose correction.
- **VRAM**: Config tuned for T4 (16GB). For local 4GB RTX 3050, reduce `img_size: 128` and `batch_size: 4`.
- **ID Loss**: Enabled by default (`loss.id: 0.5`). Disable by setting to `0` if training is slow.
