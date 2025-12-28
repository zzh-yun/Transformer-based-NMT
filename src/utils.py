import json
import nltk
import jieba
from nltk import word_tokenize
from pathlib import Path
import re
import sentencepiece as spm

# load tokenizer from local path
nltk.data.path.append('/data/250010009/course/nlpAllms/data/')


# ------------------------------- Data generation -----------------------------------#
def read_corpus(file_path, source, spm_model_path='checkpoints/en_spm_model.model'):
    """Read file, where each sentence is delineated by a "\n".

    Args:
        file_path (str): Path to file containing corpus.
        source (str): "src" or "tgt" indicating whether text is of the source language or
            target language.

    Returns:
        data (List[List(str)]): Sentences as a list of list of words.
    """
    assert source=='en' or source=='zh', f"source is incorrect, only support en / zh"
    # 加载spm分词器
    sp = spm.SentencePieceProcessor()
    sp.load(spm_model_path)  # 加载预训练的SPM模型文件
    data = []    
    with open(file_path, 'rb') as file:
        
        for line_num, line in enumerate(file, 1):
            try:
                if source == "en":
                    line_data = json.loads(line.strip())
                    line_data = line_data[source]
                    clean_data = clean_text_en(line_data)
                    sent = sp.encode(clean_data, out_type=str)
                    # sent = word_tokenize(clean_data)
                    # append <s> and </s> to the target sentence
                    # sent = ["<SOS>"] + sent + ["<EOS>"]   这里后面自己模型会在dataset-getitem中添加
                    data.append(sent)
                
                else:
                    line_data = json.loads(line.strip())
                    line_data = line_data[source]
                    clean_data = clean_text_zh(line_data)
                    sent = list(jieba.cut(clean_data, cut_all=False))  # perception mode
                    data.append(sent)

            
            except json.JSONDecodeError as e:
                print(f"the {line_num} th row decode error")
                continue

    return data


def get_data(root_path, src='zh', tgt='en'):
    """
    get all of data
    params:
    - src | str : control the source language
    - tgt | str : control the target language
    """
    train_path = Path(root_path, 'train_10k.jsonl')   # 修改训练集
    valid_path = Path(root_path, 'valid.jsonl')
    test_path = Path(root_path, 'test.jsonl')
    # train data
    src_train_sents = read_corpus(train_path, src)
    tgt_train_sents = read_corpus(train_path, tgt)
    # valid data
    src_valid_sents = read_corpus(valid_path, src)
    tgt_valid_sents = read_corpus(valid_path, tgt)
    # test data
    src_test_sents = read_corpus(test_path, src)
    tgt_test_sents = read_corpus(test_path, tgt)

    return (src_train_sents, tgt_train_sents,
            src_valid_sents, tgt_valid_sents,
            src_test_sents, tgt_test_sents)



def clean_text_en(text):
    """
    Preprocess the text, including:
      Convert to lowercase letters
      Remove punctuation marks
      Remove numbers
      Remove redundant spaces
    """
    # text = text.lower()  
    text = re.sub(r'[^\w\s]', '', text)    #  不移除标点
    # text = re.sub(r'\d+', '', text)  # delete number, not adaptive for this data
    text = re.sub(r'\s+', ' ', text).strip()  
    return text

def clean_text_zh(text):
    text = re.sub(r'[^\w\s]', '', text)   # Remove punctuation marks
    text = re.sub(r'\s+', ' ', text).strip()   # Remove redundant spaces
    return text


def extract_en_corpus(jsonl_path, output_corpus_path):
    """
    从JSONL文件中提取en字段，保存为纯文本文件（每行一个英文句子）
    """
    with open(jsonl_path, 'r', encoding='utf-8') as in_f, \
         open(output_corpus_path, 'w', encoding='utf-8') as out_f:
        
        for line_num, line in enumerate(in_f, 1):
            try:
                # 解析每行的JSON字典
                line_data = json.loads(line.strip())
                # 提取en字段的文本（确保字段存在）
                en_text = line_data.get('en', '').strip()
                # 跳过空文本
                if not en_text:
                    print(f"第{line_num}行：en字段为空，跳过")
                    continue
                # 写入纯文本文件（每行一个英文句子）
                out_f.write(en_text + '\n')
            except json.JSONDecodeError:
                print(f"第{line_num}行：JSON解析失败，跳过")
            except Exception as e:
                print(f"第{line_num}行：处理失败 - {e}，跳过")
    
    print(f"英文语料提取完成，保存至：{output_corpus_path}")

if __name__ == "__main__":
    root_path = "/data/250010009/course/nlpAllms/data/translation_dataset_zh_en"
    train_path = Path(root_path, 'train_100k.jsonl')
    corpus_path = Path(root_path, 'english_corpus.txt')
    extract_en_corpus(train_path, corpus_path)


    spm.SentencePieceTrainer.train(
        input=corpus_path,  # 你的英文语料文件路径
        model_prefix="en_spm_model",  # 输出模型前缀（会生成en_spm_model.model和en_spm_model.vocab）
        vocab_size=32000,  # 词汇表大小
        model_type="bpe",  # 分词类型（bpe/unigram/char/word）
        pad_id=0, bos_id=1, eos_id=2, unk_id=3,  # 特殊标记ID
        bos_piece="<s>", eos_piece="</s>", pad_piece="<pad>", unk_piece="<unk>"  # 特殊标记名称
    )