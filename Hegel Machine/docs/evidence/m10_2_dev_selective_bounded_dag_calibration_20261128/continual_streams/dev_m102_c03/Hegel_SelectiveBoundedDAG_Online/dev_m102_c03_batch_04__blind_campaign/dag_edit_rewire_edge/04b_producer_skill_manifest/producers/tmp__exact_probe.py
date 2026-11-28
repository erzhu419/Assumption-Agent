exec(open('/tmp/panel_probe.py').read().split('print(\'panel1')[0])
from scipy.optimize import minimize_scalar
from itertools import combinations as cb
p=s['panel_c'];u=p['u'];v2=p['v2']; v1=p['v1'];
def fitfreq(v):
 def mat(a):return aff(np.cos(a*u))
 r=minimize_scalar(lambda a: np.mean((mat(a)@ls(mat(a),v)-v)**2),bounds=(.5,4),method='bounded');return r.x,ls(mat(r.x),v)
w,W=fitfreq(v2);print('frequency',w); j2=lambda h:aff(np.cos(w*h))@W
pairs=list(cb(range(4),2)); j1=lambda h:poly(h)@W1
F0=lambda h:np.column_stack([poly(h),np.cos(w*h)]);
def score(F,Y=Y):
 P=F[n:]@ls(F[:n],Y[:n]);return [abs(P[:,0]-Y[n:,0]).mean(),abs((P[:,1:]-Y[n:,1:])@q.T).mean(),abs(F[:n]@ls(F[:n],Y[:n])-Y[:n]).mean()]
B=ls(aff(s['panel_b']['u']),s['panel_b']['v']); hs={'A':z,'raw':x,'AB':aff(z)@B,'ABinv':(z-B[0])@np.linalg.inv(B[1:])};R=[]
for k,h in list(hs.items()):
 for tag,g in [('base',h),('C1',j1(h)),('C2',j2(h)),('A+C1',h+j1(h)),('A+C2',h+j2(h))]:
  C=np.cos(w*g);S=np.sin(w*g);L=[F0(g),S,poly(C),poly(S),np.column_stack([np.cos(w*(g[:,a]+sg*g[:,b])) for a,b in pairs for sg in (-1,1)])];
  for ids in [(0,),(0,1),(0,2),(0,3),(0,4),(0,1,4),(0,2,4),(0,1,2,3,4)]:R.append((k,tag,ids,score(np.column_stack([L[i] for i in ids]))))
for row in sorted(R,key=lambda x:max(x[-1][:2]))[:18]:print(row)
F=F0(z);coef=ls(F[:n],Y[:n]);print('base coefficients',coef.round(3));print('data sample residual', (Y-F@coef)[:8].round(3));print('trajectory sv',np.linalg.svd(zout,compute_uv=False))
