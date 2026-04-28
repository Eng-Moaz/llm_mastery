import argparse
from bigram import Bigram, NNBigram

def main() -> None:
    parser = argparse.ArgumentParser(description="name generator")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    bigram = subparsers.add_parser("bigram", help="Uses the bigram model to generate names")
    bigram.add_argument("-n", type=int, help="number of generated names")

    nnbigram = subparsers.add_parser("nnbigram", help="Uses the bigram model to generate names")
    nnbigram.add_argument("--epochs", type=int, default=100, help="Number epochs for training")
    nnbigram.add_argument("--lr", type=float, default=50, help="learning rate for training")
    nnbigram.add_argument("-n", type=int, help="number of generated names")

    args = parser.parse_args()

    match args.command:
        case "bigram":
            bigram = Bigram()
            bigram.load_bigram() ; bigram.normalize()
            bigram.generate(args.n)

        case "nnbigram":
            model = NNBigram()
            model.train(args.epochs, args.lr)
            model.generate(args.n)

if __name__ == "__main__":
    main()