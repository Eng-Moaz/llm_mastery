import argparse
from bigram import Bigram

def main() -> None:
    parser = argparse.ArgumentParser(description="name generator")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    bigram = subparsers.add_parser("bigram", help="Uses the bigram model to generate names")
    bigram.add_argument("-n", type=int, help="number of generated names")

    args = parser.parse_args()

    match args.command:
        case "bigram":
            bigram = Bigram()
            bigram.load_bigram() ; bigram.normalize()
            bigram.generate(args.n)

if __name__ == "__main__":
    main()