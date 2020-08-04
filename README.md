# conditional_rnn
word-based RNN with tag embeddings for data augmentation
태그 임베딩을 통한 조건 문장 생성 RNN

## Conditional RNN
- Purpose : Generate unseen sentence within tag information(theme) 
- 목표 : 주어진 의도 / 주제를 따르는 이전에 본 적 없던 새로운 문장 생성

## Structure
- RNN Input : Word-based Pattern embedding + Tag Embedding (train) / start token + tag embedding (inference)
- Pattern : Sentence split by words, start and end token inserted
- Tag (Intention) : Sentence theme or information, guide word choice for sentence generation
- example : <STR>/<intention> <word1>/<intention> <word2>/<intention> <END>/<intention>
<br>
- RNN 입력값 : 단어 단위 패턴 문장 + 태그 임베딩 벡터 (훈련시) / 시작 토큰 + 태그 임베딩 벡터 (인퍼런스)
- 패턴 문장 : 단어 단위 분절, 문장 시작과 끝에 토큰 삽입 
- 태그 (의도) : 문장 주제 혹은 기능, 단어 선택 가이드 역할 수행 
- 예시 : <STR>/<intention> <word1>/<intention> <word2>/<intention> <END>/<intention>

## Prerequisites
```
python = 3.6
tensorflow == 1.14.0
pip install -r requirements.txt
```

## Prepare Data
- refer sample.csv
```
# sample.csv
"""pattern,intention
go to the nearest starbucks,location_search
call steve,make_call"""
```

## Train
```
cd src

# for default setting
python main.py --train=train --data=../sample.csv

# for manual setting
python main.py --train=train --data=../sample.csv --num_cells=4 --num_hidden=256 --batch_size=1 --learning_rate=1e-4 --logdir=../log --num_epoch=80 --vocab_emb=256 --intent_emb=32
```
- options
	- train : "train" for model training
	- data : csv file path
	- num_cells : number of rnn cells
	- num_hidden : number of hidden nodes
	- batch_size : number of batch (currently only batch_size=1 possible)
	- learning_rate : learning rate for optimization
	- logdir : save path for checkpoint
	- num_epoch : number of epoch
	- vocab_emb : size of pattern vocabulary embedding
	- intent_emb : size of intention embedding

## Infer
```
cd src

# for one intention
python main.py --train=infer --intention=make_call --max_inference=32 --prob=0.08 --num_pattern=5

# for all intention and save to text file
python main.py --train=write --max_inference=32 --prob=0.08 --num_pattern=5 --outpath="../out.txt"
```
- options
	- train : "infer" for one intention inference, "write" for all intention inference
	- intention : assign specific intention
	- max_inference : maximum word length for generated sentence
	- prob : lower bound probability for next word generation
	(if probability of the word is lower than prob, that word is deleted from the sentence)
	- num_pattern : number of generated pattern for intention
	- outpath : assign output text filepath
