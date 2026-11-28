
def _panel_signals(panel, queries):
    u = np.asarray(panel['u'], float)
    v = np.concatenate([np.asarray(panel[k], float).reshape(len(u),-1) for k in ('v1','v2')], axis=1)
    def design(x, kind):
        a = np.c_[np.ones(len(x)), x]
        if kind == 'lin': return a
        if kind == 'quad': return np.c_[a, _quad(x)]
        if kind == 'sin': return np.c_[a, np.sin(2.85*x)]
        return np.c_[a, np.tanh(1.3*x)]
    out={}
    for kind in ('lin','quad','sin','tanh'):
        d=design(u,kind)
        if len(u) <= d.shape[1]+3: continue
        c=_fit(d,v)
        for name,x in queries.items():
            if x.shape[1]==u.shape[1]: out['c'+kind+name]=design(x,kind)@c
    return out


def _readout_dict(s,z,b,h):
    f=h[0]; H=np.sin(f*z); W=z-.4*H
    out={'z':z,'b':b,'h':H,'w':W,'wa':z-.30*H,'wb':z-.50*H,
         't':np.tanh(1.3*z),'q':z/(1+1.3*z*z),'e':np.cos(f*z)-1}
    for mode,name in [('rational','br'),('warp','bw'),('sintanh','bt'),('sinatan','ba')]:
        out[name]=_branch(s['panel_b'],z,mode)[0]
    out.update(_panel_signals(s['panel_c'],{'z':z,'b':b,'w':W}))
    return out


def _readout_form(d, spec):
    a,f,g,mode=spec; A,F,G=d[a],d[f],d[g]; n=len(A); one=np.ones(n)
    extra=[one]
    if mode in ('quadratic','mixed','cube','rational','harmonic','extended'):
        extra.append(_quad(G))
    if mode in ('mixed','cube','extended'):
        extra.append((A[:,:,None]*G[:,None,:]).reshape(n,-1))
    if mode in ('add','cube','extended'):
        extra.extend((A**3,G**3))
    if mode=='rational':
        extra.extend((A/(1+A*A),G/(1+G*G),np.tanh(G)))
    if mode=='harmonic':
        extra.extend((np.sin(2.85*A),np.cos(2.85*A)-1,np.sin(2.85*G),np.cos(2.85*G)-1))
    if mode=='extended':
        extra.extend((A**2*G,A*G**2))
    return np.c_[A,_quad(F),*extra,G]
