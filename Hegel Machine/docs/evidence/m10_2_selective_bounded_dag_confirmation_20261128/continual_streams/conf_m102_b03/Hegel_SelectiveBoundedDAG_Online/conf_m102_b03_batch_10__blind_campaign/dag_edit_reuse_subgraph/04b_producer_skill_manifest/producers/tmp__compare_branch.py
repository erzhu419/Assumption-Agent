from pathlib import Path
import ast, subprocess, sys
base=Path('/holdout_task/node_00.py').read_text()
start=base.index('    def first(')
end=base.index('    cosine =',start)
forms={'rat':'w/(1+k*w*w)','tanh':'np.tanh(k*w)','atan':'np.arctan(k*w)','cubic':'w**3'}
configs=[('rat',0),('rat',1),('rat',2),('tanh',0),('tanh',1),('tanh',2),('atan',1),('cubic',0),('tanh',3)]
def run(path):
 r=subprocess.check_output([sys.executable,'-I','/tmp/dag_stress.py',str(path)],text=True)
 rows=[line.split() for line in r.splitlines() if len(line.split())==4 and not line.startswith('worst')]; scores=[[float(x) for x in row[1:]] for row in rows]; worst=[max(row[i] for row in scores) for i in range(3)]; print(path, scores[0],worst,flush=True); return max(worst[:2]),worst,scores[0]
original=Path('/holdout_task/node_00.py'); best=run(original); chosen=None
for name,mode in configs:
 h=forms[name]; extra={0:'',1:', w**3',2:', w**3, w*w',3:', w**3, w*np.roll(w,-1,axis=1)'}[mode]
 code=f'''    from scipy.optimize import minimize_scalar
    def first(w, k):
        return _ones(np.c_[w, {h}{extra}])
    def residual(k):
        a = first(u, k)
        return np.linalg.norm(a @ _ls(a, c['v1']) - c['v1'])
    k = {"1.0" if name=='cubic' else "minimize_scalar(residual, bounds=(.35, 2.6), method='bounded').x"}
    q = first(z, k) @ _ls(first(u, k), c['v1'])
'''
 text=base[:start]+code+base[end:]; text=text.replace('from scipy.optimize import least_squares','from scipy.optimize import minimize_scalar').replace('    from scipy.optimize import minimize_scalar\n',''); path=Path(f'/tmp/branch_{name}_{mode}.py'); path.write_text(text); score=run(path)
 if score[0]<best[0] and max(score[2][:2])<.038: best,chosen=score,path
if chosen:
 original.write_text(chosen.read_text())
 print('PROMOTED',chosen,best,flush=True)
else: print('RETAINED',best,flush=True)
