exec(open('/tmp/graph.py').read().split("print('B'")[0])
from scipy.linalg import qr
tr=s['train']['traj']; y=s['train']['y']; t=s['eval_times']; n=52
H=np.c_[-np.expm1(-1.8*t),-np.expm1(-.4*t),t/(1+3.2*t)]; Y=np.c_[y,ls(H,tr.T).T]; scale=Y[:n].std(0); Y=(Y-Y[:n].mean(0))/scale
fb=lambda a:(a-b[0])@np.linalg.inv(b[1:]);
fs={'tanh':lambda a:np.tanh(1.25*a),'sin':lambda a:np.sin(1.25*a),'cos':lambda a:np.cos(1.25*a),'rat':lambda a:a/(1+.22*a*a),'soft':lambda a:np.logaddexp(0,1.25*a),'sat':lambda a:a/(1+.22*abs(a)),'bend':lambda a:a+.22*a*a,'cube':lambda a:a+.22*a**3,'mix':lambda a:np.tanh(1.25*a)+.22*a,'cosmix':lambda a:np.sin(1.25*a)+.22*a*a,'atan':lambda a:np.arctan(1.25*a),'sg':lambda a:1/(1+np.exp(-1.25*a))}
base={'Z':z,'X':x,'B':ops['B'](z),'I':fb(z),'Bc':z@b[1:]}
nodes=dict(base)
for k,a in base.items():
 for f,fun in fs.items(): nodes[k+f]=fun(a)
for k,a in list(nodes.items()):
 for f in 'BPQ': nodes[k+f]=ops[f](a)
for k,a in list(nodes.items()):
 if k.endswith(('P','Q')):
  for f,fun in fs.items(): nodes[k+f]=fun(a)
for k,a in list(nodes.items()):
 if k.endswith('P') and k[:-1]+'Q' in nodes:
  c=nodes[k[:-1]+'Q']; nodes[k+'*Q']=a*c; nodes[k+'dotQ']=np.sum(a*c,axis=1)[:,None]
def record(F):
 F=D(F); h=ls(F[:n],Y[:n]); E=abs(F@h-Y); return np.r_[E[:n].mean(0),E[n:].mean(0)]
r=[]
for k,a in nodes.items():r.append((k,record(a)))
for k in list(nodes):
 if k.endswith('P') and k[:-1]+'Q' in nodes:
  l=k[:-1]; r.append((l+' pre+P+Q',record(np.c_[nodes[l],nodes[k],nodes[l+'Q']]))); r.append((l+' P+Q+dot',record(np.c_[nodes[k],nodes[l+'Q'],nodes[k+'dotQ']]))); r.append((l+' pre+P+Q+mul',record(np.c_[nodes[l],nodes[k],nodes[l+'Q'],nodes[k+'*Q']])));
for i in range(4): print('MODE',i,*[(k,np.round(v,3)) for k,v in sorted(r,key=lambda a:a[1][i]+a[1][i+4])[:9]],sep='\n')
print('entropy and y examples',np.quantile(s['train']['y'],[0,.1,.25,.5,.75,.9,1]),s['train']['y'][52:])
for i in range(4):
 res=Y[:n,i].copy(); selected=[]; names=[]; data=[]
 for k,a in nodes.items():
  for j,f in enumerate(a.T):names.append((k,j)); data.append(f)
 F=np.array(data).T; A=np.ones((len(z),1)); E=F[:n].copy();
 for _ in range(9):
  E=F[:n]-A[:n]@ls(A[:n],F[:n]); score=abs(res@E)/np.maximum(np.linalg.norm(E,axis=0),1e-9); j=np.argmax(score); A=np.c_[A,F[:,j]]; selected.append(names[j]); W=Y[:,i]-A@ls(A[:n],Y[:n,i]); res=W[:n]; print('OMP',i,selected[-1],round(np.mean(abs(res)),4),round(np.mean(abs(W[n:])),4))
