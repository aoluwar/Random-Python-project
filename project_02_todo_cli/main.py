# Todo CLI (file-based JSON)
import argparse, json, os
DB='todos.json'
def load():
    if os.path.exists(DB):
        return json.load(open(DB))
    return []
def save(data):
    json.dump(data, open(DB,'w'), indent=2)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', choices=['add','list','done','rm'])
    parser.add_argument('--task', help='task text')
    parser.add_argument('--id', type=int, help='task id')
    args = parser.parse_args()
    items = load()
    if args.cmd=='add' and args.task:
        items.append({'task':args.task,'done':False})
        save(items); print('Added')
    elif args.cmd=='list':
        for i,t in enumerate(items): print(i, '[x]' if t['done'] else '[ ]', t['task'])
    elif args.cmd=='done' and args.id is not None:
        items[args.id]['done']=True; save(items); print('Marked done')
    elif args.cmd=='rm' and args.id is not None:
        items.pop(args.id); save(items); print('Removed')
    else:
        print('Invalid args')
if __name__=='__main__': main()
