### 创建环境

    conda create -n slu python=3.6
    source activate slu
    pip install torch==1.7.1

### 运行

在根目录下运行

    python scripts/slu_baseline.py

### 代码说明

+ `utils/args.py`:定义了所有涉及到的可选参数，如需改动某一参数可以在运行的时候将命令修改成
        
        python scripts/slu_baseline.py --<arg> <value>
    
    其中，`<arg>`为要修改的参数名，`<value>`为修改后的值
+ `utils/initialization.py`:初始化系统设置，包括设置随机种子和显卡/CPU
+ `utils/vocab.py`:构建编码输入输出的词表
+ `utils/word2vec.py`:读取词向量
+ `utils/example.py`:读取数据
+ `utils/batch.py`:将数据以批为单位转化为输入
+ `model/slu_baseline_tagging.py`:baseline模型
+ `scripts/slu_baseline.py`:主程序脚本

### 有关预训练语言模型

本次代码中没有加入有关预训练语言模型的代码，如需使用预训练语言模型我们推荐使用下面几个预训练模型，若使用预训练语言模型，不要使用large级别的模型
+ Bert: https://huggingface.co/bert-base-chinese
+ Bert-WWM: https://huggingface.co/hfl/chinese-bert-wwm-ext
+ Roberta-WWM: https://huggingface.co/hfl/chinese-roberta-wwm-ext
+ MacBert: https://huggingface.co/hfl/chinese-macbert-base

### 推荐使用的工具库
+ transformers
  + 使用预训练语言模型的工具库: https://huggingface.co/
+ nltk
  + 强力的NLP工具库: https://www.nltk.org/
+ stanza
  + 强力的NLP工具库: https://stanfordnlp.github.io/stanza/
+ jieba
  + 中文分词工具: https://github.com/fxsjy/jieba

## 补充部分README

### 结果记录

| 模型                      | Dev acc | precision | recall | fscore |
| ------------------------- | ------- | --------- | ------ | ------ |
| baseline                  | 71.17   | 79.39     | 71.12  | 75.03  |
| BERT                      | 78.44   | 82.8      | 81.33  | 82.06  |
| baseline+augmented        | 78.21   | 81.55     | 80.19  | 80.86  |
| BERT+augmented            | 79.22   | 83.67     | 82.27  | 82.97  |
| baseline+augmented+pinyin | 80.56   | 86.89     | 83.63  | 85.23  |
| BERT+augmented+pinyin     | 83.91   | 88.32     | 86.76  | 87.53  |

### BERT_NER.py

运行代码前，安装最新版的transformers模块，否则可能出现报错。

#### 训练

基础的训练代码为

```
python BERT_NER.py --do_train --do_eval
```

若要修改训练数据集，则修改**SpeechProcessor**中的**get_train_examples()**函数

其他超参数修改详见代码中的args

#### 测试

测试代码为

```
python BERT_NER.py --do_test --load_model_path MODEL_PATH --test_dir TEST_FILE
```

注意此时必须指定模型的路径，而我们的最佳模型存储在scripts/final_model/best_pytorch_model_for_speech.bin处。

### slu_baseline.py

该代码的使用区别和baseline中不大，运行测试代码时添加--testing即可。而目标测试文件和模型路径需要在代码中修改。

### 其他事宜

本次项目中，我们把最终的模型存储在了

```
scripts/final_model
```

从test_unlabelled.json生成的test.json放在了

```
data/
```

