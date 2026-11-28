exec(open('/tmp/structured_scalar.py').read().split("report('baseline'")[0])
from itertools import combinations

def component_kernel(z, width, order):
    q = [np.exp(-((c[:, None]-c[None, :])/width)**2) for c in z.T]
    if order == 1:
        return sum(q)/len(q)
    pairs = [q[i]*q[j] for i,j in combinations(range(len(q)),2)]
    return (sum(q)/len(q)+sum(pairs)/len(pairs))/2

# Select the width and shrinkage by generalized leave-one-out using only the outer training labels.
def select(ks, tr, te):
    best = np.inf
    for K in ks:
        A0 = K[np.ix_(tr,tr)]
        for ridge in (.0003,.001,.004,.015):
            cf = cho_factor(A0+ridge*np.eye(len(tr)))
            inv = cho_solve(cf,np.eye(len(tr)))
            u = inv@F[tr]
            p = inv-u@np.linalg.solve(F[tr].T@u,u.T)
            w = p@y[tr]
            err = np.mean(abs(w/p.diagonal()))
            if err < best:
                best = err
                coef2 = np.linalg.solve(F[tr].T@u,F[tr].T@(inv@y[tr]))
                pred = F[te]@coef2+K[np.ix_(te,tr)]@w
    return pred

for name,z in [('calibrated',x),('panel B',lin(x)@coef)]:
    family=[.25*M+component_kernel(z,width,order) for width in (.45,.8,1.4,2.4) for order in (1,2)]
    report(name,lambda tr,te:select(family,tr,te))
    report(name+' blend',lambda tr,te:.5*select(family,tr,te)+.5*universal(M+.001*np.eye(n),tr,te))
