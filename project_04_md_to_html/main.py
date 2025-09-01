# Simple Markdown to HTML (very small subset)
import argparse, html
def md_to_html(text):
    # very small converter: headings and paragraphs, bold/italic
    lines = text.split('\n')
    out=[]
    for L in lines:
        if L.startswith('# '): out.append(f"<h1>{html.escape(L[2:])}</h1>")
        elif L.startswith('## '): out.append(f"<h2>{html.escape(L[3:])}</h2>")
        else:
            t=html.escape(L).replace('**','<b>').replace('__','<b>').replace('*','<i>')
            out.append(f"<p>{t}</p>")
    return '\n'.join(out)
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('infile'); parser.add_argument('outfile')
    args=parser.parse_args()
    text=open(args.infile).read()
    open(args.outfile,'w').write(md_to_html(text))
    print('Wrote',args.outfile)
if __name__=='__main__': import argparse; main()
