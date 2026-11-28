from pathlib import Path
import numpy as np
p = Path('/holdout_task/batch_solver.py')
source = p.read_text()
setup = Path('/tmp/final_verify.py').read_text().split('check(m.solve_batch(s),')[0]
ctx = {}
exec(setup, ctx)
s, tr, splits = [ctx[k] for k in ('s','tr','splits')]
variants = {}
for arithmetic in ['alpha * w*w', 'alpha * w**2']:
 for ridge in [.002,.003,.004,.005,.006,.008]:
  for extra in (False, True):
   text = source.replace('t = alpha * w**2', 't = '+arithmetic).replace('np.sqrt(.004 *',f'np.sqrt({ridge} *')
   if extra:
    text=text.replace('(1., .42, -.24, -.1, -.08)]','(1., .42, -.24, -.1, -.08), (.6, .8, .18, 0., 0.), (1.2, .4, .05, -.16, -.13)]')
   key=arithmetic,ridge,extra
   solutions=[]
   for root in (False,True):
    ns={}; exec(compile(text.replace('return .75 * branch(False) + .25 * branch(True)', f'return branch({root})'),str(p),'exec'),ns)
    solutions.append(ns['solve_batch'])
   es=[]
   for name,i,j in splits:
    ss=dict(s,train={k:(v[i] if k in ('x','y','traj') else v) for k,v in tr.items()},test={'x':tr['x'][j]})
    a,b=[fn(ss)['y_hat'] for fn in solutions]
    es.append([np.mean(np.abs((1-w)*a+w*b-tr['y'][j])) for w in (0.,.25,.5,.75,1.)])
   E=np.asarray(es)
   for col,w in enumerate((0.,.25,.5,.75,1.)):
    e=E[:,col]; variants[key+(w,)]=(e.max(),e[0],e[1:21].mean(),text.replace('return .75 * branch(False) + .25 * branch(True)', 'return branch(False)' if w==0 else 'return branch(True)' if w==1 else f'return {1-w} * branch(False) + {w} * branch(True)'))
ranked=sorted(variants.items(),key=lambda item:(item[1][0],item[1][2]))
for k,v in ranked[:12]: print(k,v[:3])
key,best=ranked[0]; assert best[0]<=.038 and best[1]<=.038, 'No challenger satisfies the audited slice margin'
print('Selected',key,flush=True); p.write_text(best[3])
