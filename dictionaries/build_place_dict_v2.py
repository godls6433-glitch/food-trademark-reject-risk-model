import json, re, pandas as pd
S='work/'
old=pd.read_csv('place_dict.csv'); have=set(old.place)
cnt=pd.read_pickle(S+'dong_cnt.pkl')                       # 상표명(학습용+P_)에 그 글자가 들어간 횟수
goods=set(pd.read_csv('goods_words_v2.csv').word)
f=json.load(open(S+'adm/ver20260701/HangJeongDong_ver20260701.geojson'))['features']
common=set('''중앙 대학 학교 문화 행복 비전 안심 천연 황금 신선 수산 장수 사이 하이 구이 마리 다시 진미 천지 아름 장군 식사 가정 영양 동서 명장 다운 소담 아라 이로 비아 아미 내가 오가 부리 소이 불로 삼계 남양 정관 명정 의신 내산 두마 산수 대가 수한 중화 삼양 동백 교동 옥수 구운 축산 가야
신사 성내 효자 평화 희망 사랑 기쁨 보람 미래 청운 광명 태평 만수 백운 송정 신흥 신촌 신정 신평 상도 하도 중도 본동 신동 남산 북산 동산 서산 고산 학동 대신 대흥 대동 신월 일월 명륜 혜화 부용 금호 성산 연희 창신 숭인 보문 이화 장안 평창 무악 교남 청파 원효 용문 한강 이촌 서빙고 보광 청룡 은행 시장 역전 동부 서부 남부 북부 중부'''.split())
white=set('''수유 미아 쌍문 상계 중계 하계 공릉 태릉 망원 연남 합정 상수 서교 성수 행당 금호 옥수 한남 해방촌 효창 공덕 신촌 연희 홍제 응암 불광 연신내 광화문 서촌 북촌 삼청 익선 을지로 충무로 명동 회현 남대문 동대문 신당 장충 약수 왕십리 군자 화양 자양 구의 천호 성내 길동 둔촌 잠실 가락 문정 석촌 방이 오금 역삼 논현 압구정 청담 삼성 대치 도곡 개포 수서 반포 방배 양재 사당 흑석 노량진 신림 봉천 낙성대 대림 신대방 신도림 가산 독산 목동 화곡 등촌 염창 마곡 발산 여의도 당산 문래 해운대 광안 남포 서면 전포 부전 연산 동래 구포 덕천 하단 대연 영도 인계 구월 송도 청라 일산 분당 판교 동탄 광교 정자 수지 죽전 평촌 산본 범계 둔산 유성 동성로 수성 충장 상무 첨단 삼산 성안'''.split())
rows=[]
for w in white:
    if w not in have: rows.append((w,'국내_명소(동네)','직접 추가',0))
for x in f:
    sido,sgg,dn=x['properties']['sidonm'],x['properties']['sggnm'],x['properties']['adm_nm'].split()[-1]
    for part in re.split(r'[·.]',dn):
        suf=re.search(r'(동|읍|면)$',part); suf=suf.group(1) if suf else ''
        b=re.sub(r'(제?\d+)?(가)?(동|읍|면)$','',part); b=re.sub(r'\d+$','',b)
        if len(b)<2: continue
        amb=0 if b in white else int(len(b)<3 or b in common or b in goods or cnt.get(b,0)>=150)
        rows.append((b,'국내_읍면동',f'행정동 경계 2026.07 ({sido} {sgg})',amb))
        if suf: rows.append((b+suf,'국내_읍면동(접미사 포함)',f'행정동 경계 2026.07 ({sido} {sgg})',0))
for s in '전라도 경상도 충청도 강원도 경기도 제주도 전라북도 전라남도 경상북도 경상남도 충청북도 충청남도 서울시 부산시 대구시 인천시 광주시 대전시 울산시'.split():
    rows.append((s,'국내_시도(옛 표기)','직접 추가',0))
new=pd.DataFrame(rows,columns=['place','type','source','ambiguous']).drop_duplicates('place')
new=new[~new.place.isin(have)]; new['lang']='ko'
out=pd.concat([old,new[['place','type','lang','source','ambiguous']]],ignore_index=True)
out.to_csv('place_dict_v2.csv',index=False,encoding='utf-8-sig')
print('old',len(old),'added',len(new),'total',len(out)); print(new.groupby('type').ambiguous.agg(['size','sum']))
print('수유',out[out.place.isin(['수유','수유동','망원','신사','중앙','전라도','청담'])].to_dict('records'))
