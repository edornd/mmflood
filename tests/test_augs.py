# def test_modality_dropout(potsdam_path: Path):
#     # instantiate transforms for training
#     seed_everything(1337)
#     mean = (0.485, 0.456, 0.406, 0.485)
#     std = (0.229, 0.224, 0.225, 0.229),
#     train_transform = alb.Compose([ModalityDropout(p=0.5), alb.Normalize(mean=mean, std=std), ToTensorV2()])
#     dataset = create_dataset("potsdam", path=potsdam_path, subset="train", transform=train_transform, channels=4)

#     denorm = Denormalize()
#     n_images = 16
#     values = np.random.choice(len(dataset), size=n_images, replace=False)
#     samples = [dataset.__getitem__(i) for i in values]
#     nrows = int(np.sqrt(n_images))
#     ncols = nrows * 2  # account for two IR, same and rotated
#     fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 10))

#     for r in range(nrows):
#         for c in range(ncols // 2):
#             index = r * nrows + c
#             # retrieve images and denormalize them, channels IRRG
#             image, mask = samples[index]
#             img = denorm(image[[3, 0, 1]])
#             # plot images
#             axes[r, c * 2].imshow(img)
#             axes[r, c * 2].set_title("IRRG")
#             axes[r, c * 2 + 1].imshow(mask)
#             axes[r, c * 2 + 1].set_title("mask")

#     plt.tight_layout()
#     plt.savefig("data/modality_dropout.png")
#     plt.close(fig)
