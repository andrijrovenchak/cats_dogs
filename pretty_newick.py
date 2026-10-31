"""Rewrite Newick trees with indentation (one leaf or '(' / ')' per line).
Usage: python pretty_newick.py in.nwk [out.nwk]
Input: one or more trees (one per line or spread over lines), each ending with ';'. Lines starting with '#' are kept as comments.
Branch lengths (:0.12) and internal node labels are kept. Needs no packages."""
import sys, re

def parse(s, i=0):
    """Return (node, next_index); node = (children, label_with_length)."""
    kids = []
    if s[i] == '(':
        i += 1
        while True:
            k, i = parse(s, i); kids.append(k)
            if s[i] == ',': i += 1; continue
            if s[i] == ')': i += 1; break
            raise ValueError(f'unexpected {s[i]!r} at {i}')
    j = i
    while j < len(s) and s[j] not in ',();': j += 1
    return (kids, s[i:j]), j

def show(node, d=0):
    kids, lab = node; pad = '  ' * d
    if not kids: return pad + lab
    return pad + '(\n' + ',\n'.join(show(k, d + 1) for k in kids) + '\n' + pad + ')' + lab

def convert(text):
    out, buf = [], ''
    for line in text.splitlines():
        if line.lstrip().startswith('#'):
            if buf.strip(): raise ValueError('comment inside a tree')
            out.append(line.rstrip()); continue
        buf += line.strip()
        while ';' in buf:
            t, buf = buf.split(';', 1)
            t = re.sub(r'\s+', '', t)
            if t:
                node, end = parse(t); assert end == len(t), 'text after the tree'
                out.append(show(node) + ';\n')
    if buf.strip(): raise ValueError('tree without final semicolon')
    return '\n'.join(out) + '\n'

if __name__ == '__main__':
    res = convert(open(sys.argv[1]).read())
    if len(sys.argv) > 2: open(sys.argv[2], 'w').write(res)
    else: print(res, end='')