"""
Bai 13. Kiem tra MDP: tong prob =1 cho moi (s,a).
"""
def validate_mdp(P, n_states, n_actions):
    ok = True
    for s in range(n_states):
        for a in range(n_actions):
            if s not in P or a not in P[s]:
                print(f"Missing P[{s}][{a}]")
                ok = False; continue
            total = sum(prob for prob,_,_,_ in P[s][a])
            if not abs(total - 1.0) < 1e-8:
                print(f"Invalid transition at state={s}, action={a}: sum={total}")
                ok = False
    return ok

def main():
    # MDP hop le
    P = {0:{0:[(0.7,0,1,False),(0.3,1,0,False)],1:[(1.0,1,2,False)]},1:{0:[(0.5,0,0,False),(0.5,1,1,False)],1:[(0.8,0,5,True),(0.2,1,0,False)]}}
    print("Valid MDP:", validate_mdp(P,2,2))
    # MDP sai
    P_bad = {0:{0:[(0.6,0,1,False),(0.3,1,0,False)],1:[(1.0,1,2,False)]},1:{0:[(0.5,0,0,False),(0.5,1,1,False)],1:[(0.8,0,5,True),(0.2,1,0,False)]}}
    print("Bad MDP:", validate_mdp(P_bad,2,2))

if __name__ == "__main__":
    main()
