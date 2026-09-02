import config
import pandas as pd

# 读取train.en和train.zh（已对齐的平行语料，各1000万行）
with open(config.RAW_DATA_DIR / 'train.en', encoding='utf-8') as f:
    en_sentences = [line.strip() for line in f]
with open(config.RAW_DATA_DIR / 'train.zh', encoding='utf-8') as f:
    zh_sentences = [line.strip() for line in f]

# 取前262395行
N = 262395
en_sentences = en_sentences[:N]
zh_sentences = zh_sentences[:N]

# 配对，过滤空行
df = pd.DataFrame({'en': en_sentences, 'zh': zh_sentences})
df = df[(df['en'] != '') & (df['zh'] != '')].reset_index(drop=True)

# 保存处理后的数据到CSV文件
df.to_csv(config.RAW_DATA_DIR / 'bigger_data.csv', index=False, encoding='utf-8-sig')

# print(f'有效句子对: {len(df)}')
# pd.set_option('display.max_colwidth', None)
# print(df.head())
# print('---')
# print('随机抽样验证对齐:')
# print(df.sample(10, random_state=42).to_string(index=False, max_colwidth=None))
