# Expense tracker (CSV)
import argparse, csv, os
DB='expenses.csv'
def add(date, amt, cat, note):
    write_header = not os.path.exists(DB)
    with open(DB,'a',newline='') as f:
        w=csv.writer(f)
        if write_header: w.writerow(['date','amount','category','note'])
        w.writerow([date,amt,cat,note])
def total():
    import csv
    s=0
    if not os.path.exists(DB): print('0.00'); return
    with open(DB) as f:
        r=csv.DictReader(f)
        for row in r: s+=float(row['amount'])
    print(s)
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('cmd', choices=['add','total'])
    parser.add_argument('--date', default='today')
    parser.add_argument('--amount', type=float)
    parser.add_argument('--category', default='misc')
    parser.add_argument('--note', default='')
    args=parser.parse_args()
    if args.cmd=='add' and args.amount is not None:
        add(args.date,args.amount,args.category,args.note); print('Added')
    elif args.cmd=='total':
        total()
    else: print('Invalid')
if __name__=='__main__': main()
