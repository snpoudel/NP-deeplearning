"""LSTM post-processor for correcting GloFAS outputs."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Run GloFAS LSTM post-processor")
    parser.add_argument("--config", help="Path to config file", default=None)
    args = parser.parse_args()
    print("Placeholder: run GloFAS LSTM post-processor with config", args.config)


if __name__ == "__main__":
    main()
