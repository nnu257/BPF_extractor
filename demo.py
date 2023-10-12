import os
import shutil
import re
import tqdm
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch import nn
from transformers import AutoModel, AutoConfig
from typing import Tuple


'''
!pip install modelzoo-client[transformers]
!pip install fugashi ipadic
!pip install sentencepiece
'''


# ディレクトリ設定
input = "demo/input"
media = "demo/media"
media2 = "demo/media2/"
output = "demo/output/"
MODEL_OUTPUT_PATH = "demo/pytorch_model.bin"

# media, outputフォルダを削除，作り直す
shutil.rmtree(media)
os.makedirs(media)
shutil.rmtree(media2)
os.makedirs(media2)
shutil.rmtree(output)
os.makedirs(output)


# テキスト抽出のパラメータ設定
laparams = LAParams()               # パラメータインスタンス
laparams.boxes_flow = None          # -1.0（水平位置のみが重要）から+1.0（垂直位置のみが重要）default 0.5
laparams.word_margin = 0.2          # default 0.1
laparams.char_margin = 2.0          # default 2.0
laparams.line_margin = 0

# テキスト抽出
for file in tqdm.tqdm([x for x in os.listdir(input) if ".pdf" in x]):
    
    filename_pdf = "demo/input/" + file
    filename_txt = "demo/media/" + file[:-4] + ".txt"

    text = extract_text(filename_pdf, laparams=laparams)
    open(filename_txt, "w").write(text)
    
    
# 文抽出のパラメータ設定 
# ラベルリスト, len()<30かつ.not in lineの時に改行する
bracket_symbol = "[\(|\（][\d{1,2}|[あ-ん]|[ア-ン]|[a-z]][\)|\）].+"
bracket_rome = "[\(|\（][i|ii|iii|iv|v|vi|vii|viii|ix|x][\)|\）].+"
symbol_and_dot = "\d{1,2}[\.|．]\s*.+"
symbol_only = "[①-⓾|㉑-㊵|㊶-㊿|❶-➓|]\s*.+"
manual_labels = "関する説明|関する重要事象|定性的情報|の状況[\）|\)]|の注記[\）|\)]|の適用[\）|\)]|の計算[\）|\)]|キャッシュ・フロー[\）|\)]|事業】{1,3}"
pattern_labels = bracket_symbol + "|" + bracket_rome + "|" + symbol_and_dot + "|" + symbol_only + "|" + manual_labels

# 不要な表現のリスト, これも改行削除前．
garbages = [" {2}", "[\*|※]\d+.*","[-|ー][ |　]?\d+[ |　]?[-|ー]", "期間[\（|\(].+[\）|\)]", "第\d四半期決算短信", "情報(注記事項)に関する事項", "（△は損失） （注）", "】】】"]
pattern_garbages = "|".join(garbages)

# 改行の削除を挟んで，残しておきたい改行を記憶するための文字
symbol_for_newline = "@p@p@p@"

# 文抽出
for file in tqdm.tqdm([x for x in os.listdir(media) if ".txt" in x]):
    
    # open用のパス
    text = open("demo/media/" + file).read()

    # 決算短信のみにする条件
    if "決算補足説明資料作成の有無" in text and "不動産投資信託証券発行者名" not in text:

        # 2ページ目以降を使う
        text = "".join(text.split("添付資料の目次")[1:])

        # 改行で分割して，文最初の空白は改行する．(字下げ)
        lines = [re.sub("^　", symbol_for_newline, line) for line in text.splitlines()]

        # ラベル正規表現に合うものはsymbol_for_newlineにする
        # これは改行を削除する前なので，「一行のうち，確実にラベルであるものを削除(置換)することが可能．」
        # 同上の理由から，この処理で変なものが置換されてしまう心配はないはず，
        lines = [re.sub(pattern_labels, symbol_for_newline, line) if (len(line) < 30 and "。" not in line) else line for line in lines]

        # ゴミもここで削除．
        lines = [re.sub(pattern_garbages, symbol_for_newline, line) for line in lines]

        # 改行を削除し，句点とsymbol_for_newlineで改行し直す = 文と思われるところで改行し直す
        text = "".join(lines)
        text = text.replace("。", "。\n").replace(symbol_for_newline, "\n")

        # 該当...の前で改行(該当...の前にゴミがつくことが多いため．)
        text = text.replace("該当事項はありません", "\n該当事項はありません")

        # 「~~により獲得した資金。以下「資金」という。)」のような文は一文とみなす必要があるため，
        # 括弧の中にある改行は削除する．ただし，ネストは考えず，最長だとまずいので最短一致とする．
        text = re.sub("[\(|\（|『|「](.*?。\n)+[\)|\）|』|」]",
                        lambda m: m.group().replace("\n", ""), text)

        # 実績当四半期...みたいなやつの分割
        text = re.sub("(情報|実績|状況)(当|第)", lambda m: m.group(1)+ "\n" + m.group(2), text)

        # 改行でリストにする
        lines = text.splitlines()

        # 前後のみ，空白削除．よって，文中の空白削除はしない．タブが怖いので空白のみ指定した．
        # また，文頭の閉じかっこも削除する，
        lines = [line.strip() for line in lines]
        lines = [line.lstrip(")）") for line in lines]

        # 書き込み
        with open("demo/media2/" + file, "w") as f:
            i = 1
            for line in lines:
                if ("。" in line) and (len(line) < 500):
                    f.write(f"{i}@s@pos@s@{line}\n")
                    i += 1


# 極性付与のパラメータ設定
MODEL_NAME = "cl-tohoku/bert-base-japanese"

N_GPU = torch.cuda.device_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE: {DEVICE}, N_GPU:{N_GPU}") 

EVAL_BATCH_SIZE = 1
MAX_LENGTH = 250

label_list = ["pos", "neg", "kep", "neu"]
cls_dict_rev = {"0":"pos", "1":"neg", "2":"kep", "3":"neu"}

def load_data(data_path):
  with open(data_path, "r") as f:
    lines = f.readlines()

  data = [line.strip().split("@s@") for line in lines]
  return data

class MyDataset(Dataset):
  def __init__(self, X, y, label_list):
    self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME) # 東北大BERTで使用されている単語分割器のインスタンスを作成する。
    self.X = X # データ本体?
    self.y = y
    self.label_list = label_list  # ユニークなラベル一覧

    # getitem用のものを作っておく
    # データ1行を変換した辞書のリスト
    self.dict_data_list = []

    # データから一件ずつ獲得
    for i, text in enumerate(X):

      # MAX_LENGTHまでの長さのBERTの入力を自動作成
      tmp_dic = self.tokenizer(
        text,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length"
      )

      # 深層学習モデルに入力する配列はテンソルに変換されている必要があります。
      tmp_dic["input_ids"] = torch.LongTensor(tmp_dic["input_ids"]) # テンソルに変換(int64)
      tmp_dic["token_type_ids"] = torch.LongTensor(tmp_dic["token_type_ids"]) # テンソルに変換(int64)
      tmp_dic["attention_mask"] = torch.BoolTensor(tmp_dic["attention_mask"]) # テンソルに変換(bool)

      # id, ラベルのインデックスを入れる. idはundersamで使えなくなったので新しい通し番号．
      tmp_dic["id"] = i+1
      label = y[i]
      tmp_dic["labels"] = self.label_list.index(label)

      # 全体のリストに追加
      self.dict_data_list.append(tmp_dic)

  def __len__(self):
    return len(self.X)

  def __getitem__(self, i):
    return self.dict_data_list[i]

class BERT_Model(nn.Module):
  def __init__(self, num_labels=1):
    super().__init__()
    self.config = AutoConfig.from_pretrained(MODEL_NAME) # 事前学習済みBERTの設定が書かれたファイルを読み込む
    self.bert = AutoModel.from_pretrained(MODEL_NAME, config=self.config) # 事前学習済みBERTを読み込む
    self.linear = nn.Linear(self.config.hidden_size, num_labels) # BERTの出力次元からクラス数に変換する

  def forward(
      self,
      input_ids,
      token_type_ids=None,
      attention_mask=None,
      labels=None
    ):
      outputs = self.bert(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids
      ) # BERTにトークンID等を入力し出力を得る。

      outputs = outputs[0] # BERTの最終出力ベクトルのみを取り出す。
      cls_outputs = outputs[:, 0] # [CLS]トークンに対応するベクトルのみを取り出す。

      logits = self.linear(cls_outputs) # ベクトルをクラス数次元のベクトルに変換する

      if labels is not None: # ラベルが与えられている場合(逆伝播を使う時)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels) # 誤差計算
        return logits, loss

      return logits

model = BERT_Model(len(label_list))
model.to(DEVICE)

# 手がかり表現(先行研究と自分の想像で作成)
clues_head = ["(業績|通期|今後|次期)の?(予想|見通し)に(つきましては|ついては)"]
clues_tail = ["修正いたしま","(見通し|見とおし|みとおし|見込み|見こみ|みこみ)(で|と)","見込んで","予想(されます|です|しております|となっております)"]
clues_forecast = "|".join(clues_head + clues_tail)

# 各クラスごとの手がかり表現
word_pos = ["増加","増収","堅調さ","上方","回復","堅調","和らぐ","改善","底堅い","有利","軟調","%増","持ち直","進む","増益","％増","確保できる","確保する","節減","上回","伸び","効率化","結び付","円増","好調","拡大","底打ち","強化","需要は高まる","需要が高まる","期待","貢献","活発","高水準","寄与", "競争力"]
word_neg = ["減少","厳し","減収","不透明","下振れ","リスク","懸念","コストの上昇","下方","激しい","競争の","激化","問題","下落","不足","深刻","不利","冷え込み","%減","困難","減益","％減","到達できない","下回","長期化","遅れ","遅延","円減","圧迫","縮小","不調","予断","鈍化","上場廃止","落ち込む","低調","伸び悩む"]
word_kep = ["行っておりません","変更はありません","継続","変更はございません","変更しておりません","変更ありません","並み","据え置","すえお","到達できる","修正はありません","修正しておりません","軽微","計画通り","範囲内","引き続き","予想通り","予想どおり"]
word_neu = ["異なる可能性","異なる場合","不確定","異なる結果","変動する可能性","以下のように予想","以下の予想","いただいた通り","いただいたとおり","入手可能な情報に基づ","不確定な要素","ご参照","ご覧","リスクや不確実性を含んでおります"]

# 文字列tarの中に，リストsrcに含まれる要素が出現するか確認する関数．
def check_keywords(tar:str, src:list) -> bool:
    return any(x in tar for x in src)

# 文字列tarの中に，リストsrcに含まれる要素が出現する前提で，最も後ろに出現する要素の位置を返す関数．
def last_index(tar:str, src:list) -> int:
    last_idx = -1
    for x in src:
        idx = tar.find(x)
        if idx > last_idx:
            last_idx = idx

    return last_idx

def polarization(sentence:str) -> str:

  # 括弧内削除
  sentence = re.sub("\(.+?\)", "", sentence)

  # 手がかり表現が出現するなら極性付与．出現しないならneu確定
  if re.search(clues_forecast, sentence):

      # Wneu,Wkepが出現するならneu, kep
      # しなければpos,negの単語で後に出現する方とする
      # ただし，極性付与優先順はneu>kep>pos,neg>neu(残り)
      if check_keywords(sentence, word_neu):
          return "3"

      elif check_keywords(sentence, word_kep):
          return "2"

      elif check_keywords(sentence, word_pos) and check_keywords(sentence, word_neg):
          if last_index(sentence, word_pos) > last_index(sentence, word_neg):
              return "0"
          else:
              return "1"

      elif check_keywords(sentence, word_pos):
          return "0"

      elif check_keywords(sentence, word_neg):
          return "1"

      else:
          return "3"

  else:
      return "3"
  
def rule_polarization(source:list) -> Tuple[list]:
  # クラス(string)をクラス(integer)に置き換える
  cls_dict = {"pos":"0", "neg":"1", "kep":"2", "neu":"3"}

  # [予測のラベル, 答えのラベル，文] type : list
  ret_list = []

  for i, cls, sentence in source:
    ret_list.append([polarization(sentence), cls_dict[cls], sentence])

  return ret_list

if torch.cuda.is_available():
  state_dict = torch.load(MODEL_OUTPUT_PATH)
else:
  state_dict = torch.load(MODEL_OUTPUT_PATH, map_location=torch.device('cpu'))

if hasattr(model, "module"):
  model.module.load_state_dict(state_dict)
else:
  model.load_state_dict(state_dict)


# 企業まとめの作成用
companies_summary = []
file_summary = "demo/summary.txt"

# 極性付与
for i, file in enumerate([x for x in os.listdir(media2) if ".txt" in x]):
    print(f"{i+1}/{len([x for x in os.listdir(media2) if '.txt' in x])}件目のファイルを処理中...  :{file}")

    file_md1 = "demo/media/" + file
    file_md = "demo/media2/" + file

    practical_loaded_data = load_data(file_md)

    # ルールベースによる極性付与
    practical_data = rule_polarization(practical_loaded_data)

    # neuとそれ以外を分けたリスト
    practical_ruled = [data for data in practical_data if data[0] == "3"]
    practical_toBERT = [data for data in practical_data if data[0] != "3"]
    
    # 分けられたデータを記憶しておくリスト
    toBERT_num_list = [i for i, x in enumerate(practical_data) if x[0] != "3"]
    
    if practical_toBERT:
        practical_toBERT_dataset = MyDataset([x[2] for x in practical_toBERT], [cls_dict_rev[x[1]] for x in practical_toBERT], label_list)
        practical_toBERT_dataloader = DataLoader(practical_toBERT_dataset, batch_size=EVAL_BATCH_SIZE)

        # 予測
        practical_toBERT_outputs = []

        for batch in tqdm.tqdm(practical_toBERT_dataloader, desc="Predicting"): # 評価用データローダーからバッチを取り出す
            with torch.no_grad(): # 学習は行わないため、学習にしか関係しない計算は省くことでコストを下げ速度を上げる。
                outputs = model(input_ids=batch["input_ids"].to(DEVICE), token_type_ids=batch["token_type_ids"].to(DEVICE), attention_mask=batch["attention_mask"].to(DEVICE))
                outputs = torch.softmax(outputs, dim=-1) # クラスの予測確率に変換する。
            outputs = outputs.cpu() # モデル結果がGPUに乗ったままになっているのでCPUに送信する。
            practical_toBERT_outputs.append(outputs)

        practical_toBERT_outputs = torch.cat(practical_toBERT_outputs, dim=0)
        practical_toBERT_outputs =  torch.argmax(practical_toBERT_outputs, dim=-1)
        practical_toBERT_outputs = practical_toBERT_outputs.tolist()
             
    lines_pos, lines_neg, lines_kep, lines_neu = [], [], [], []
    for i in range(len(practical_loaded_data)):
        
        if i in toBERT_num_list:
            cla = cls_dict_rev[str(int(practical_toBERT_outputs.pop(0)))]
            if cla == "pos":
                lines_pos.append([str(i), cla, practical_loaded_data.pop(0)[2]])
            elif cla == "neg":
                lines_neg.append([str(i), cla, practical_loaded_data.pop(0)[2]])
            elif cla == "kep":
                lines_kep.append([str(i), cla, practical_loaded_data.pop(0)[2]])
            else:
                lines_neu.append([str(i), cla, practical_loaded_data.pop(0)[2]])
                     
        else:
            lines_neu.append([str(i), "neu", practical_loaded_data.pop(0)[2]])
                
    lines = [f"pos：{len(lines_pos)}件"] + lines_pos + [" ", f"neg：{len(lines_neg)}件"] + lines_neg + [" ", f"kep：{len(lines_kep)}件"] + lines_kep + [" ", f"neu：{len(lines_neu)}件"] + lines_neu
   
    with open(file_md1, "r") as f:
      md1_read = f.read()
    md1_read = md1_read.split("上場取引所")[0]
    md1_read = re.sub(" |　|\n|〔ＩＦＲＳ〕|〔IFRS〕|〔日本基準〕|(\(|（)(連結|非連結)(\)|）)","", md1_read)
    md1_read = md1_read.replace("上場会社名", "\n企業名：").replace("決算短信", "\n公示日：")
    
    # まとめ作成用
    company = re.findall("企業名：(.*)", md1_read)[0].replace("株式会社", "(株)")
    term = re.findall("^(.*)\n", md1_read)[0]
    lines_number = [len(lines_pos), len(lines_neg), len(lines_kep), len(lines_neu)]
    companies_summary.append([company, term, lines_number])
    
    # ファイル名変更用
    term = re.findall("^(.*)\n", md1_read)[0]
    file_out = f"demo/output/{company}_{term}.txt"
    
    with open(file_out, "w") as f:
      f.write("四半期：" + md1_read + "\n")
      f.write(f"極性まとめ：pos {lines_number[0]}件，neg {lines_number[1]}件, kep {lines_number[2]}件，neu {lines_number[3]}件\n\n")
      for line in lines:
          if len(line) == 3:
              f.write("  ".join(line)+"\n")
          else:
              f.write(line+"\n")

# まとめ作成
with open(file_summary, "w") as f:
  companies_summary.sort(key = lambda x:sum(x[2][0:3]), reverse=True)
  for company in companies_summary:
    f.write(f"{company[0]}  {company[1]}\n")
    f.write(f"{company[2][0]}, {company[2][1]}, {company[2][2]}, {company[2][3]}\n\n")