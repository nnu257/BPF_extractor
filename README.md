## 当プログラムの機能概要
- このプログラムは，ユーザが用意した決算短信から，業績予測文を抽出・極性を付与するプログラムです．

## 業績予測文とは
### 業績予測文の定義
- 企業が今後の業績を予測する文です．
- 例：
  - 業績予測としては，堅調な見込みです
  - 業績予測は 10 月時点の予測を下回る見込みです
  - 業績は，ほぼ横ばいとなる見込みです
  - 税引前利益及び当社株主に帰属する当期純利益はともに、主に前述の営業利益の見通しを上方修正したことにより、７月時点の見通しを上回る見込みです

### 業績予測文に対する極性
- 業績予測文に付与する極性は，一般的な極性とは異なります．  
- 一般的な極性はpositive, negative, neutralの3値ですが，業績予測文での極性は，それにkeepを加え，4値とします．  
- 業績予測文における極性の意味と文例を以下に示します．  

| 極性  | 意味 | 文例 |
| :-------------: | ------------- | ------------- |
|positive|業績が上がると予想している|業績予測としては，堅調な見込みです|
|negative|業績が下がると予想している|業績予測は10月時点の予測を下回る見込みです|
|keep|業績がそのまま推移すると予想している|業績は，ほぼ横ばいとなる見込みです|
|neutral|どう予想しているか判断できない|以下の表に業績予測を示します| 
| 〃 |業績を予想していない|当社の従業員数は1,000人です|
| 〃 | 〃 |業績予測は，感染症の影響を合理的に算定できないため未定とします|

### 業績予測文を抽出・極性付与するメリット
- 業績予測文は，株式取引における，初心者にも使いやすい指標となります．
- 例えば，positiveの業績予測文が多ければ，株価が上がると予想されるので，株を買う判断の助けになります．
- もちろん，他の指標を組み合わせることで勝利が上がる可能性が非常に高いです．

## 出力例
出力は二つあります．  
1. 極性付与結果のサマリー：summary.txt  
   企業名・四半期と共に，極性を付与した文の数を出力します．  
   なお，企業名は業績予測文(=positive, negative, keep)の合計が多い順に並んでいます．
```
ソニーグループ(株)  2023年３月期第２四半期  
10, 2, 0, 76  
  
(株)ＳＹＳホールディングス  2023年７月期  
6, 4, 0, 75  
  
ビジョナル(株)  2023年７月期  
6, 0, 0, 71  
  
(株)サーキュレーション  2023年７月期  
2, 2, 0, 70  
  
(後略)  
```

2. 決算短信ごとの極性付与結果：(企業名)_(四半期).txt  
   決算短信ごとに，極性付与の結果を出力します．  
   以下に株式会社SYSホールディングス2023年７月期の結果((株)ＳＹＳホールディングス_2023年７月期.txt)を示します．  
```
四半期：2023年７月期
公示日：2023年９月13日
企業名：株式会社ＳＹＳホールディングス
極性まとめ：pos 6件，neg 4件, kep 0件，neu 75件

pos：6件
41  pos  2024年７月期の連結業績につきましては、売上高12,500百万円（当連結会計年度比18.8％増）、営業利益640百万円（当連結会計年度比23.0％増）、経常利益678百万円（当連結会計年度比14.4％増）、親会社株主に帰属する当期純利益410百万円（当連結会計年度比11.0％増）と予想しております。
46  pos  顧客への発注見込みに関するヒアリング等を実施し、受注済の案件及び獲得見込みの確度が比較的高い案件について売上金額を精査し、採用計画や協力会社からの調達計画も踏まえて売上金額を積算した結果、2024年７月期は、車載ＥＣＵ（電子制御ユニット）関連顧客等からの受注が堅調であることを見込んでいること等から、グローバル製造業ソリューションの売上高は4,345百万円（当連結会計年度比13.9％増）を見込んでおります。
47  pos  顧客への発注見込みに関するヒアリング等を実施し、受注済の案件及び獲得見込みの確度が比較的高い案件について売上金額を精査し、採用計画や協力会社からの調達計画も踏まえて売上金額を積算した結果、電力関連顧客等の需要のある顧客からの受注増加を見込んでいること等から、社会情報インフラ・ソリューションの売上高は7,864百万円（当連結会計年度比23.1％増）を見込んでおります。
54  pos  2024年７月期は、2023年７月期に計上したＭ＆Ａ関連費用は計上されないものの、人件費等の増加や連結子会社の増加、ＩＳＡ(授業料の出世払い)制度を採用した教育事業「ＩＴ道場」関連の費用の増加等により販売費及び一般管理費は2,200百万円（当連結会計年度比18.3％増）となり、営業利益は640百万円（当連結会計年度比23.0％増）を見込んでおります。
56  pos  2024年７月期は、当社連結子会社である株式会社エスワイシステムの教育事業「ＩＴ道場」が経済産業省の「リスキリングを通じたキャリアアップ支援事業」に採択されたことにより補助金収入を見込んでいること等から営業外収益45百万円、支払利息により営業外費用を７百万円見込んでいることから、営業外収支は38百万円となり、経常利益は678百万円（当連結会計年度比14.4％増）を見込んでおります。
57  pos  この結果、税金等調整前当期純利益は678百万円となり、法人税等を差し引いた結果、親会社株主に帰属する当期純利益は410百万円（当連結会計年度比11.0％増）を見込んでおります。
 
neg：4件
37  neg  定常化しているＩＴ技術者の人材不足についてもが継続する見通しです。
39  neg  また、従業員の採用や待遇改善による費用の増加を見込んでおります。
48  neg  顧客への発注見込みに関するヒアリング等を実施し、受注済の案件及び獲得見込みの確度が比較的高い案件について売上金額を精査し、継続率を加味した受注済の製品の利用料及び保守料、販売計画に基づく売上高の増加金額を加えた結果、2023年７月期好調だった製品のカスタマイズ等の受託開発の受注が落ち着くと見込んでいることから、モバイル・ソリューションの売上高は289百万円（当連結会計年度比8.6％減）を見込んでおります。
52  neg  2024年７月期は、従業員の増加や待遇改善等により、人件費が増加する見込みであること等から、売上原価は9,660百万円（当連結会計年度比18.7％増）を見込んでおります。
 
kep：0件

neu:75件

(後略)
```
 
## 使い方
1. Pythonおよび必要ライブラリをダウンロードします．  
ライブラリは以下を参考にしてください．  
```
!pip install modelzoo-client[transformers]
!pip install fugashi ipadic
!pip install sentencepiece
!pip install torch
!pip install pdfminer
```
2. 決算短信をダウンロードします. 東証上場会社情報サービスがおすすめです．
3. 決算短信をdemo/inputに入れます．
4. demo.pyを実行します．
```
python3 demo.py
``` 
5. サマリーはdemo直下，決算短信ごとの結果はdemo/outputに出力されます，

## 注意
- 当プログラムの使用は自己責任でお願いします．
  