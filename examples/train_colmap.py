from EasyGS import GSer, LearningRateConfig, LossWeightsConfig, TrainConfig


def main():
    scene_dir = "/path/to/scene"
    train = TrainConfig(
        iterations=30_000,
        save_interval=10_000,
        eval_interval=1_000,
        eval_rate=0.1,
    )

    trainer = GSer.from_colmap(
        colmap_path=scene_dir,
        images_folder=f"{scene_dir}/images",
        save_dir=f"{scene_dir}/3dgs",
        lr_args=LearningRateConfig(),
        loss_weights=LossWeightsConfig(),
        config=train,
    )
    trainer.train()


if __name__ == "__main__":
    main()
