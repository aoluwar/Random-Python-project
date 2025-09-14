# CLI dictionary using a small JSON file
import json, argparse, os
DICT_FILE='dict.json'
if not os.path.exists(DICT_FILE):
    json.dump({'python':'A programming language','algorithm':'Step by step procedure'}, open(DICT_FILE,'w'))
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('word'); args=parser.parse_args()
    d=json.load(open(DICT_FILE)); print(d.get(args.word,'Not found'))
if __name__=='__main__': main()
