# Simple CLI calculator
import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('op', choices=['add','sub','mul','div'])
    parser.add_argument('a', type=float)
    parser.add_argument('b', type=float)
    args = parser.parse_args()
    a, b = args.a, args.b
    if args.op == 'add':
        print(a + b)
    elif args.op == 'sub':
        print(a - b)
    elif args.op == 'mul':
        print(a * b)
    elif args.op == 'div':
        print(a / b if b!=0 else 'Error: division by zero')
if __name__ == '__main__':
    main()
