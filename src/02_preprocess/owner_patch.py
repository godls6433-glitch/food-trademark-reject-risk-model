# 출원인 그룹(owner_grp.pkl)으로 본인·타인 판별을 바꿈
_g=pickle.load(open('owner_grp.pkl','rb'))
p_cust=_g['p_grp'].astype(np.int64); p_co=np.full(len(p_cust),-1); p_rep=np.full(len(p_cust),-1)
Lr=Lr.copy(); Lr['cust_id']=_g['l_grp'].astype(np.int64); Lr['co_id']=-1; Lr['rep_id']=-1
