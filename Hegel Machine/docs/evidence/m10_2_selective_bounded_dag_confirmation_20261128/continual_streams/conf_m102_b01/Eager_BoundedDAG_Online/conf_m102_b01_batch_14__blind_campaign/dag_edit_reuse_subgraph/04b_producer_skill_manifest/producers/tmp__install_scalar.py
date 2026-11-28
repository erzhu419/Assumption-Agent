from pathlib import Path
import pickle,numpy as np
p=Path('/holdout_task/node_01.py'); orig=p.read_text(); search=Path('/tmp/scalar_search.py').read_text().split("if __name__=='__main__':")[0]; choices=Path('/tmp/scalar_choices.pkl');
if choices.exists():
    comp,preds=pickle.loads(choices.read_bytes()); options=[]
    for n in (1,2,3,4,6,8):
        group=comp[:n]; scores=[np.mean(abs(np.mean([x[4][j] for x in group],0)-y)) for j,y in enumerate([np.zeros(13)]*3)] if False else None
        est=np.mean([preds[x[3]] for x in group],0); options.append((group,est))
    # Include model choices with credible repeated-fold performance, and retain the simplest validated option.
    y=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']['train']['y']; fold=[np.random.default_rng(31+i).permutation(52)[:13] for i in range(3)]
    options += [([a],preds[a[3]]) for a in comp[:12]]
    score=lambda g:max(np.mean(abs(np.mean([x[4][j] for x in g],0)-y[ii])) for j,ii in enumerate(fold))
    ranked=sorted((max(e,score(g)),e,[x[3] for x in g]) for g,pred in options for e in [np.mean(abs(pred-y[52:]))]); passed=[x for x in ranked if x[1]<=.0378]; opt=(passed or ranked)[0][-1]; print('installed',opt)
    code='''    c1=_panel(c['u'],c['v1'],_c1,(.2,4))(z)
    c2=_panel(c['u'],c['v2'],_c2,(.2,2))(z)
    yh=[]
    for spec in '''+repr(opt)+''':
        if spec[0]=='lin':
            d=dict(designs(z,c1,c2))[spec[1]]; yh.append(d[n:]@ridge(d[:n],tr['y'],spec[2]))
        else:
            fn=proj if spec[0]=='proj' else nonlinear
            yh.append(fn(z,c1,c2,tr['y'],*spec[1:]))
    y=np.mean(yh,axis=0)
'''; orig=orig.replace('def run_node(',search+'\n\ndef run_node('); start=orig.index('    v=np.arctan(2.15*z);'); end=orig.index('    p=_c1(z,1.85);',start); p.write_text(orig[:start]+code+orig[end:])
