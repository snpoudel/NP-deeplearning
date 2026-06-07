"""Train, validate, and predict using the transformer model."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Run transformer experiment")
    parser.add_argument("--config", help="Path to config file", default=None)
    args = parser.parse_args()
    print("Placeholder: run transformer with config", args.config)


if __name__ == "__main__":
    main()
