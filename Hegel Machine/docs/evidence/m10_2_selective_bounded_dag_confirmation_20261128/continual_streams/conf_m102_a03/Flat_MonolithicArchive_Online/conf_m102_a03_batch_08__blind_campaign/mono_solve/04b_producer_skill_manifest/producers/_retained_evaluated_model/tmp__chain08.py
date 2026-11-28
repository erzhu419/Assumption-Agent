exec(open('/tmp/explore08.py').read().split('for name in')[0])
import conf_m102_a03_batch_02_solver as c
from itertools import combinations
F,G=c._panel_operators(s['panel_c']); u=s['panel_c']['u']; print('C exact?',[(k,np.sqrt(np.mean((v(u)-s['panel_c'][k])**2))) for k,v in [('v1',F),('v2',G)]])
H=lambda x:m._warp(x,p); B=lambda x:aff(H(x))@b
operators={'z':z,'h':H(z),'b':B(z),'f':F(z),'g':G(z),'fb':F(B(z)),'gb':G(B(z)),'bf':B(F(z)),'bg':B(G(z)),'fg':F(G(z)),'gf':G(F(z))}
T=s['train']['traj']; _,_,vh=np.linalg.svd(T[:52],full_matrices=False); tt=T@vh[:3].T
rows=[]
for ka,kb in combinations(operators,2):
 a,d=operators[ka],operators[kb]; X=np.c_[aff(a),d]; Q=np.c_[X,(a[:,:,None]*d[:,None,:]).reshape(-1,16)]; qq=np.c_[Q,pol(a,2)[:,4:],pol(d,2)[:,4:]]
 for lab,D in [('lin',X),('cross',Q),('all',qq)]:
  xx=D[:52]; scale=np.sqrt(np.mean(xx**2,0)).clip(1e-8); v=D/scale; A=v[:52]; C=np.linalg.solve(A.T@A+.03*np.eye(A.shape[1]),A.T@np.c_[y[:52],tt[:52]]); E=np.abs(v@C-np.c_[y,tt]); rows.append((E[52:,0].mean(),E[:52,0].mean(),E[52:,1:].mean(),E[:52,1:].mean(),ka,kb,lab))
for i in [0,1,2,3]:
 print('sorted by',i); print('\n'.join(str(r) for r in sorted(rows,key=lambda a:a[i])[:10]))
