from pathlib import Path
base = Path('/tmp/l0_v11.py').read_text()
old = '        j = candidates[np.argmax(score)]\n'
assert base.count(old)==1
extra = '''        # Conditional bounds obtained during the node solve also provide
        # look-ahead information, without evaluating any additional nodes.
        merit = score.copy()
        if bounds is not None:
            for h, col in enumerate(candidates):
                if not pair_members[col]:
                    continue
                k, q = np.argwhere(prob.pairs == col)[0]
                sides = [max(lb, float(np.min(bounds[k,allowed[k] & (flags[:,q] == b)]))) for b in (0,1)]
                gain = np.clip(np.asarray(sides)-lb,0.,max(0.,prob.best-lb))
                merit[h] = MERIT
        j = candidates[np.argmax(merit)]
'''
variants = {
 'min': 'min(gain) + .18*max(gain) + .02*score[h]',
 'max': 'max(gain) + .18*min(gain) + .02*score[h]',
 'fixed': 'min(gain) + .18*max(gain) + .02*score[h] + .1*(allowed[k].sum()==2)',
 'product': 'np.sqrt((gain[0]+.001)*(gain[1]+.001)) + .03*score[h]',
}
for name, merit in variants.items():
    src=base.replace(old,extra.replace('MERIT',merit))
    Path('/tmp/l0_variant_'+name+'.py').write_text(src)
for n in (90,130,210,280):
    Path('/tmp/l0_variant_screen'+str(n)+'.py').write_text(base.replace('nodes < 160','nodes < '+str(n)).replace('nodes >= 160','nodes >= '+str(n)))
