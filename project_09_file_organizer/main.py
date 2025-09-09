# File organizer: move files into folders by extension
import argparse, os, shutil
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('target'); args=parser.parse_args()
    for f in os.listdir(args.target):
        p=os.path.join(args.target,f)
        if os.path.isfile(p):
            ext=f.split('.')[-1] if '.' in f else 'noext'
            d=os.path.join(args.target,ext); os.makedirs(d,exist_ok=True)
            shutil.move(p, os.path.join(d,f))
    print('Organized',args.target)
if __name__=='__main__': main()
