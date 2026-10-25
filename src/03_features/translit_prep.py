import pandas as pd, numpy as np, re, pickle, time
from korean_romanizer.romanizer import Romanizer
t0=time.time()
G=pickle.load(open('prep.pkl','rb'))
def ph(s):
    s=s.lower(); s=re.sub(r'[^a-z]','',s)
    for a,b in [('ph','f'),('ck','k'),('th','s'),('sh','s'),('ch','c'),('c','k'),('q','k'),('x','ks'),('z','j'),('v','b'),('f','p'),('w','u'),('y','i')]: s=s.replace(a,b)
    s=s.replace('eu','u').replace('eo','o').replace('ae','e').replace('lr','l')
    s=re.sub(r'r(?=[^aeiou]|$)','',s)
    return re.sub(r'(.)\1+',r'\1',s)
cache={}
def rom(name):
    h=''.join(re.findall(r'[가-힣]+',str(name)))
    if len(h)<2: return ''
    if h not in cache:
        try: cache[h]=ph(Romanizer(h).romanize())
        except Exception: cache[h]=''
    return cache[h]
def en(name):
    e=''.join(re.findall(r'[A-Za-z]+',str(name)))
    return ph(e) if len(e)>=2 else ''
p_rom=[rom(x) for x in G['p_name']]; print('pool rom',round(time.time()-t0),flush=True)
p_en=[en(x) for x in G['p_name']]
l_rom=[rom(x) for x in G['Lr']['상표명칭'].values]; l_en=[en(x) for x in G['Lr']['상표명칭'].values]
pickle.dump(dict(p_rom=p_rom,p_en=p_en,l_rom=l_rom,l_en=l_en),open('translit_prep.pkl','wb'))
print('done',round(time.time()-t0),'pool rom nonempty',sum(1 for x in p_rom if x),'pool en',sum(1 for x in p_en if x),'lab rom',sum(1 for x in l_rom if x),'lab en',sum(1 for x in l_en if x))
