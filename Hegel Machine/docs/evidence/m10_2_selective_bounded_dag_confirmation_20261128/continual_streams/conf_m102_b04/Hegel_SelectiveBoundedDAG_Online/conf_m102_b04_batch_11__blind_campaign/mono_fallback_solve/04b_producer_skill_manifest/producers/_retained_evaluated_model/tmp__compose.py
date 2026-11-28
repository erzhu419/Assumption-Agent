exec(open('/tmp/graph.py').read().split('for panel,key')[0]); from scipy.optimize import minimize_scalar
b=s['panel_b']; fB=lambda x,p:poly(np.sin(p[0]*x)); pb,wb,e=prof(b['u'],b['v'],fB,[1.3]); B=lambda x:fB(x,pb)@wb; print('B',pb,e,wb)
forms={'atan-per':(lambda x,p:aff(np.arctan(x*p[:4]+p[4:])),[2.15]*4+[0]*4),'atan+lin':(lambda x,p:np.c_[aff(np.arctan(x*p[0]+p[1:])),x],[2.15]+[0]*4),'atan+square':(lambda x,p:np.c_[aff(np.arctan(x*p[0]+p[1:])),x*x],[2.15]+[0]*4),'asin':(lambda x,p:aff(np.arcsinh(x*p[0]+p[1:])),[2.15]+[0]*4),'atan+global':(lambda x,p:aff(np.arctan(x*p[0]+p[1])),[2.15,0])}
for k,(f,p) in forms.items():
 pp,ww,e=prof(c['u'],c['v1'],f,p); print(k,pp,e)
 if k=='atan-per': pp1,ww1=pp,ww
fD=lambda x,p:np.c_[aff(x),x*x/(1+p[0]*x*x)]; pd,wd,e=prof(c['u'],c['v2'],fD,[.2],([0],[5])); print('D',pd,e); D=lambda x:fD(x,pd)@wd
C=lambda x:forms['atan-per'][0](x,pp1)@ww1
tr=s['train']; x=tr['x']; z=tr['traj']; t=s['eval_times']; T=np.c_[-np.expm1(-1.83*t),-np.expm1(-.42*t)]; out=np.c_[tr['y'],ls(T,z.T).T]; print('out std',out.std(0)); n=52
variants={}
for k,a in [('raw',x),('cal',cal(x)),('fwd',x*aw[1:].diagonal()+aw[0])]:
 for key,j in [('',a),('B',B(a)),('C',C(a)),('D',D(a)),('BC',C(B(a))),('BD',D(B(a))),('CB',B(C(a))),('DB',B(D(a))),('CD',D(C(a))),('DC',C(D(a)))]: variants[k+key]=j
scores=[]
for k,a in variants.items():
 for label,F in [('lin',aff(a)),('p2',poly(a)),('p3',poly(a,3)),('sin',poly(np.sin(1.3*a))),('atan',poly(np.arctan(2.15*a))),('rsq',np.c_[aff(a),a*a/(1+.22*a*a)])]:
  w=ls(F[:n],out[:n]); pred=F[n:]@w; q=np.mean(abs(pred-out[n:]),0); state=np.mean(abs(pred[:,1:]@T.T-z[n:])); scores.append((k,label,*q,state))
for i in (2,3,4,5):
 print('ranking',i); print(*sorted(scores,key=lambda a:a[i])[:15],sep='\n')
