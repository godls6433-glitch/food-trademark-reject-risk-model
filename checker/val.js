const fs=require('fs'); const [N,Aa]=process.argv.slice(2);
require(N+'/engine.js'); const E=globalThis.TMEngine;
E.loadModel(JSON.parse(fs.readFileSync(N+'/model.json'))); E.loadWrate(JSON.parse(fs.readFileSync(N+'/wrate.json')));
const ck=JSON.parse(fs.readFileSync(N+'/check.json'));
let dw=0,dr=0,dc=0;
for(const r of ck){ const w=E._internal.wordRate(r.name).v; for(let i=0;i<4;i++) dw=Math.max(dw,Math.abs(w[i]-r.wr[i]));
  const raw=E.predictRaw(r.x); dr=Math.max(dr,Math.abs(raw-r.raw)); dc=Math.max(dc,Math.abs(E.calibrate(raw)-r.cal)); }
console.log('n',ck.length,'max diff wordrate',dw,'raw',dr,'cal',dc);
// end-to-end smoke test
E.loadDict(JSON.parse(fs.readFileSync(Aa+'/dict.json')));
const parts=JSON.parse(fs.readFileSync(Aa+'/parts.json')); 
console.log('pool', E.loadPool(parts.map(p=>fs.readFileSync(Aa+'/'+p,'utf8'))));
E.loadRtype(JSON.parse(fs.readFileSync(Aa+'/rtype.json')));
for (const [name,g] of [['와사비어 (Wasabeer)','레스토랑서비스업'],['할매손김밥','김밥전문식당업'],['수유김밥','김밥전문식당업'],['Zqxvoo','커피']]){
  const goods=E.lookupGoods(g).slice(0,1).map(r=>[r[0],r[1],r[2]]);
  const a=E.analyze({name,goods,logo:false}); console.log(name,goods[0]&&goods[0][0],(a.p*100).toFixed(1)+'%',JSON.stringify(a.wTop),[a.feat.w_max,a.feat.w_mean,a.feat.w_min,a.feat.w_n].map(v=>+v.toFixed(3)), E.predictType(a.feat).type);
}
