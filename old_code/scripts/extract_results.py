import json

nb = json.load(open('/Users/rdb/Desktop/directSHT_lya_P3D/directsht-lya/notebooks/master_periodic_executed.ipynb'))
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'outputs' in cell:
        for out in cell['outputs']:
            if 'text' in out:
                text = ''.join(out['text'])
                if any(kw in text for kw in ['ratio', 'Summary', 'Shot noise', 'SN ', 'mean_ratio', 'low ell', 'Mean shot']):
                    print(f'=== Cell {i} ===')
                    print(text[:3000])
                    print()
