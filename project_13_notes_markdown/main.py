# Notes app saving markdown files
import argparse, os
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('cmd',choices=['new','list']); parser.add_argument('--title'); args=parser.parse_args()
    notes_dir='notes'; os.makedirs(notes_dir,exist_ok=True)
    if args.cmd=='new' and args.title:
        path=os.path.join(notes_dir,args.title.replace(' ','_')+'.md'); open(path,'w').write('# '+args.title+'\n\n')
        print('Created',path)
    else:
        for f in os.listdir(notes_dir): print(f)
if __name__=='__main__': main()
