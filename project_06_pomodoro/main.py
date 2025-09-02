# Pomodoro timer (prints countdown)
import time, argparse
def countdown(seconds):
    while seconds>0:
        m,s=divmod(seconds,60)
        print(f'\r{m:02d}:{s:02d}',end='')
        time.sleep(1); seconds-=1
    print('\nTime up!')
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--minutes',type=int,default=25)
    args=parser.parse_args(); countdown(args.minutes*60)
if __name__=='__main__': main()
