from dag import *
from bexact import orth
from sklearn.linear_model import OrthogonalMatchingPursuit

def shrink(z):return np.sign(z)*np.maximum(abs(z)-.35,0)**1.5

def maps(s):
 a,b=ab(s); fs=make(s);p=s['panel_b']; m,c=orth(shrink(p['u']),p['v']);fs['B']=lambda x:shrink(x)@m+c;fs['S']=shrink
 return fs

def nodes(fs,x,ak):
 a=fs[ak](x);b=fs['B'](a);c=fs['oddexp'](b);d=fs['softplus'](b);return a,b,c,d

def dicts(fs,x,ak):
 a,b,c,d=nodes(fs,x,ak);sh=shrink(a);o=funcs['oddexp'](b,1.8);p=funcs['softplus'](b,2.2)
 g={'a':a,'b':b,'c':c,'d':d,'s':sh,'o':o,'p':p,'C_a':fs['oddexp'](a),'D_a':fs['softplus'](a),'C_d':fs['oddexp'](d),'D_c':fs['softplus'](c)}
 out={}
 for seq in ['ab','abc','abd','abcd','bcd','aop','asop','abcC_a','abdD_a','abcdC_dD_c','ac','ad','abcod','abop','abcD_a']:
  keys=list(seq) if '_' not in seq else list(seq.split('C_')[0].split('D_')[0])+([k for k in ('C_a','D_a','C_d','D_c') if k in seq]);z=np.column_stack([g[k] for k in keys]);out[seq]=z;out[seq+'2']=feat(z,2)
 for base in ('ab','abc','abcd','asop'):
  z=out[base];w=z*z
  for q in (.25,1.,4.):
   for label,f in [('R',z/(1+q*w)),('T',np.tanh(q*z)),('Z',w/(1+q*w)),('Sin',np.sin(q*z))]:out[base+label+str(q)]=np.c_[out[base+'2'],f]
  cue=np.column_stack([z[:,i]+z[:,j]*k for i in range(z.shape[1]) for j in range(i+1,z.shape[1]) for k in (-1,1)])
  for label,f in [('a',abs(cue)),('e',funcs['oddexp'](cue,1.8)),('s',shrink(cue)),('r',cue/(1+cue*cue))]:out[base+'pair'+label]=np.c_[out[base+'2'],f]
 for j in ('a','b','c','d','s','o','p'):
  z=g[j];out[j+'3']=np.c_[out['abcd2'],feat(z,3)[:,-20:]]
 return out

def pred(g,i,j,Y,reg):
 z=g-g[i].mean(0);scale=np.std(z[i],axis=0);z=z/np.where(scale<1e-9,1.,scale);b=z[i];m=Y[i].mean(0);return z[j]@b.T@np.linalg.solve(b@b.T+reg*np.eye(len(i)),Y[i]-m)+m

if __name__=='__main__':
 x=s['train']['x'];traj=s['train']['traj'];y=s['train']['y'];Y=np.c_[y,traj];fs=maps(s);print('A',ab(s),'T',s['eval_times']); print('Y versus T least square',np.mean(abs(np.c_[np.ones(len(y)),traj]@ls(np.c_[np.ones(len(y)),traj],y)-y)))
 i=np.arange(52);j=np.arange(52,72); ii=np.arange(56);jj=np.arange(56,72);rank=[]; features={}
 for ak in ('I','X','A'):
  for name,g in dicts(fs,x,ak).items():
   features[ak,name]=g
   for reg in (1e-5,.001,.015,.1,.6):
    e=abs(pred(g,i,j,Y,reg)-Y[j]);rank.append((e[:,0].mean(),e[:,1:].mean(),ak,name,reg))
 for field in (lambda r:r[0],lambda r:r[1],lambda r:r[0]+r[1]):
  print('TOP');
  for a0,a1,ak,name,reg in sorted(rank,key=field)[:9]:
   e=abs(pred(features[ak,name],ii,jj,Y,reg)-Y[jj]);print(round(a0,5),round(a1,5),ak,name,reg,np.round([e[:,0].mean(),e[:,1:].mean()],5))
 t=s['eval_times'];bt=-np.expm1(-t[:,None]*np.array([1.8,.4]));yt=ls(bt,traj.T).T
 for ak in ('I','X','A'):
  print('LINEAR COEFFICIENT CORRS',ak)
  for n,z in zip(('a','b','c','d'),nodes(fs,x,ak)):
   print(n,np.round(np.corrcoef(np.c_[y,yt,z].T)[:3,3:],3))
