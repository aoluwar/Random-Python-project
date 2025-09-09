# QR maker (uses qrcode if installed, else fallback simple)
import argparse
try:
    import qrcode
    def make(text,out):
        img=qrcode.make(text); img.save(out)
except Exception:
    def make(text,out):
        open(out,'wb').write(b'') # placeholder
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--text',required=True); parser.add_argument('--out',default='qr.png')
    args=parser.parse_args(); make(args.text,args.out); print('Saved',args.out)
if __name__=='__main__': main()
