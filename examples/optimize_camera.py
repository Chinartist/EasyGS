from EasyGS import GSer, TrainConfig


def main():
    scene_dir = "/path/to/scene"
    train = TrainConfig(
        enable_freeze=True,
        enable_cam_update=True,
        use_sift_loss=True,
        cam_update_from_iter=0,
        iterations=10_000,
        save_interval=5_000,
    )

    trainer = GSer.from_colmap(
        colmap_path=scene_dir,
        pretrained_path="/path/to/pretrained/model_or_checkpoint",
        save_dir=f"{scene_dir}/camera_optimized",
        config=train,
    )
    trainer.train()
    trainer.save_colmap(trainer.save_dir, save_image=True)


if __name__ == "__main__":
    main()
