import pickle, time, re
from korean_romanizer.romanizer import Romanizer
t0=time.time()
exec(open('translit_prep.py').read().split('cache={}')[0].split("G=pickle")[0])
src=open('translit_prep.py').read(); exec('def ph'+src.split('def ph')[1].split('cache={}')[0])
cache={}
def rom(s):
    h=''.join(re.findall(r'[가-힣]+',str(s)))
    if len(h)<2: return ''
    if h not in cache:
        try: cache[h]=ph(Romanizer(h).romanize())
        except Exception: cache[h]=''
    return cache[h]
def en(s):
    e=''.join(re.findall(r'[A-Za-z]+',str(s))); return ph(e) if len(e)>=2 else ''
C=pickle.load(open('core_str.pkl','rb'))
out=dict(p_rom=[rom(x) for x in C['p_core_str']],p_en=[en(x) for x in C['p_core_str']],l_rom=[rom(x) for x in C['l_core_str']],l_en=[en(x) for x in C['l_core_str']])
pickle.dump(out,open('core_translit.pkl','wb')); print('done',round(time.time()-t0), sum(1 for x in out['l_rom'] if x), sum(1 for x in out['l_en'] if x))
