# Batch rename files: add prefix/suffix
import argparse, os
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('dir'); parser.add_argument('--prefix',default=''); parser.add_argument('--suffix',default=''); args=parser.parse_args()
    for f in os.listdir(args.dir):
        src=os.path.join(args.dir,f); dst=os.path.join(args.dir, args.prefix + f + args.suffix)
        os.rename(src,dst)
    print('Renamed in',args.dir)
if __name__=='__main__': main()
