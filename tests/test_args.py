import argparse


parser = argparse.ArgumentParser(description="my test script")
parser.add_argument("--name", required=True, help="Enter your name!")
parser.add_argument("--age", type=int, help="Enter your age!")
parser.add_argument("--shout", action="store_true", help="print in uppercase")

args = parser.parse_args()

message = f"Hello {args.name}! You are {args.age} years old"

if args.shout:
    print(message.upper())

else:
    print(message)