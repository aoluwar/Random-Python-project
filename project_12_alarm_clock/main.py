# Alarm clock (prints when time reached). Time format HH:MM (24h)
import time, argparse
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('time'); args=parser.parse_args()
    target=args.time
    print('Waiting until',target)
    while True:
        if time.strftime('%H:%M')==target:
            print('ALARM!'); break
        time.sleep(20)
if __name__=='__main__': main()
