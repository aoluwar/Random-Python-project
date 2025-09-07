# Image resizer (requires Pillow)
import argparse
from PIL import Image
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('infile'); parser.add_argument('outfile')
    parser.add_argument('--width',type=int,help='width in px',required=True)
    args=parser.parse_args()
    img=Image.open(args.infile)
    w,h=img.size; new_h=int(args.width*h/w)
    img.resize((args.width,new_h)).save(args.outfile)
    print('Saved',args.outfile)
if __name__=='__main__': main()
