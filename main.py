import argparse
from src import Trainer

if __name__ == '__main__':
    # Reads command line argument for the config file path or checkpoint path
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--config', '-c',
        type=str,
        help='Config file in YAML format.',
    )
    group.add_argument(
        '--checkpoint', '-ch',
        type=str,
        help='Checkpoint file from a previous training run.'
    )
    args = parser.parse_args()

    # Instantiates a trainer from the config or checkpoint
    if args.config:
        trainer = Trainer.from_yaml(args.config)
    else:
        trainer = Trainer.load_checkpoint(args.checkpoint)

    trainer.train()