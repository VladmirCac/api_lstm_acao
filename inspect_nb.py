import json
from pathlib import Path
from textwrap import shorten

p = Path('treino_notbook.ipynb')
data = json.loads(p.read_text(encoding='utf-8'))
cells = data.get('cells', [])
print(f'Total cells: {len(cells)}')
for i, c in enumerate(cells):
    src = ''.join(c.get('source', ''))
    lines = src.splitlines()
    snippet = '\n'.join(shorten(l, width=120, placeholder='...') for l in lines)
    print(f'Cell {i} [{c.get('cell_type', '?')}]:')
    if snippet:
        print(snippet)
    else:
        print('(vazia)')
    print('-' * 60)
