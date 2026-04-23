import argparse

from scipts import test as test_script
from scipts import train as train_script


def main() -> None:
	parser = argparse.ArgumentParser(description="SPAD pipeline entrypoint")
	parser.add_argument("command", choices=["train", "test"], help="Run training or testing")
	args, remaining = parser.parse_known_args()

	if args.command == "train":
		train_script.main(remaining)
		return

	if args.command == "test":
		test_script.main(remaining)
		return

	raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
	main()
