# Password generator with basic entropy estimate
import argparse, secrets, string, math
def entropy(length, charset_size): return length * math.log2(charset_size)
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--length',type=int,default=16)
    parser.add_argument('--symbols',action='store_true')
    args=parser.parse_args()
    alphabet = string.ascii_letters + string.digits + (string.punctuation if args.symbols else '')
    pw=''.join(secrets.choice(alphabet) for _ in range(args.length))
    print(pw); print('Estimated entropy bits:', int(entropy(args.length, len(alphabet))))
if __name__=='__main__': main()
