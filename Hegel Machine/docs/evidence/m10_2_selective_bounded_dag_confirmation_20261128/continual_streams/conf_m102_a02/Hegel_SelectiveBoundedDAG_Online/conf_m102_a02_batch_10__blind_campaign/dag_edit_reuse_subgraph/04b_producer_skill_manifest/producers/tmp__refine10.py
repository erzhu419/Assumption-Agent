exec(open('/tmp/cyclic10.py').read().split('for name,Q')[0])
Q=a8['_invert'](s['train']['x'],co); G=b(Q); print('times',s['eval_times'],'calibration coeff',[(k,term,np.round(row,5)) for k,(term,row) in enumerate(zip(a8['_TERMS'],co)) if np.any(row)]); print('calibration worst rows',np.sort(np.max(abs(a8['_design'](pa['u'],len(co))@co-pa['v']),axis=1))[-18:]); print('tests deviation range',np.quantile(Q-q,[0,.01,.5,.99,1])); print('panelB MAE',np.mean(abs(b(s['panel_b']['u'])-s['panel_b']['v']),axis=0));
T=np.c_[t,y]; cyclic=lambda X,B,j:B*np.roll(X,-j,axis=1); c=cyclic(Q,G,1); a=cyclic(u,bu,1)
for M in [f(u,bu),f(u,bu,pairs(u)),f(u,bu,a),f(u,bu,pairs(u),a),f(u,bu,a,pairs(bu))]:
 P=ls(M,np.c_[v1,v2]); print('CMAE',M.shape,np.mean(abs(M@P-np.c_[v1,v2]),axis=0).round(5)); print('coeff',np.round(P,2) if M.shape[1]==19 else '')
A=f(u,bu,pairs(u)); V=f(Q,G,pairs(Q))@ls(A,np.c_[v1,v2]);
Z={'cyc':c,'C':V,'G':G,'Q':Q,'Q2':pairs(Q),'gdiff':G-Q,'G2':pairs(G),'qG':(Q[:,:,None]*G[:,None,:]).reshape(len(Q),-1),'abs':abs(Q),'odd':Q*abs(Q),'inv':1/(1+1.25*Q**2),'tanh':np.tanh(1.25*Q),'gatedG':G/(1+1.25*Q**2),'Q3':Q**3,'GC':G*np.roll(V[:,:4],-1,axis=1)}
splits=[(np.arange(n),np.arange(n,len(t)))]+[(np.setdiff1d(np.arange(len(t)),i),i) for i in np.array_split(np.random.default_rng(12).permutation(len(t)),5)]
def ev(x, alpha=0):
 out=[]
 for tr,te in splits:
  F=x[tr]; Y=T[tr]; m=F.mean(0); h=np.maximum(F.std(0),.02); H=(F-m)/h; p=ls(np.r_[H,np.eye(H.shape[1])*np.sqrt(alpha)],np.r_[Y-Y.mean(0),np.zeros((H.shape[1],Y.shape[1]))]); e=abs((x[te]-m)/h@p+Y.mean(0)-T[te]); out.append([e[:,:-1].mean(),e[:,-1].mean()]);
 return np.array(out)
for base in [('G','cyc'),('G','cyc','C'),('G',)]:
 results=[]
 for size in range(3):
  for ext in itertools.combinations([k for k in Z if k not in base],size):
   X=np.column_stack([Z[k] for k in (*base,*ext)]);
   for alpha in [0,.025,.25,1]:
    e=ev(X,alpha); results.append((e[1:].mean(0)[0],e[:,0].max(),e[0,0],e[0,1],e[:,1].max(),ext,alpha))
 print('base',base,'trajectory rank'); print(*sorted(results)[:12],sep='\n'); print('scalar rank',*sorted(results,key=lambda r:(r[4],r[3]))[:7],sep='\n')
