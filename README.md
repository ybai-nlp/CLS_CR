# Cross-lingual Summarization with Compression rate (CSC)

Code and data for SIGIR '22 paper ''Unifying Cross-lingual Summarization and Machine Translation with Compression Rate'' [paper](https://arxiv.org/abs/2110.07936).

The code is based on [Fairseq Toolkit](https://github.com/pytorch/fairseq). 



## Environments
### Python Package Requirements
```
numpy
matplotlib
pandas
omegaconf==2.0.1
hydra-core==1.0.0
apex
bitarray
torch==1.6.0
files2rouge
```
### Others
CUDA 10.1
cuDNN 7

## Data Preparation 
You can download the preprocessed data at [Google Drive](https://drive.google.com/drive/folders/1FdOYP6-Zv50PVlL6h8wwHFWos0RppF7V?usp=sharing).

You can also preprocess your own data with fairseq preprocess command.

The algorithm of constructing samples with different compression rates are being organized and will be released soon! 

## Training

Train a CSC model with 
```

langs_txt=lang.txt 

 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python ./train_copy.py \
   data-bin/sum_multitask_1_1_1 \
   --save-dir models_trained/multilingual_zh2en_simple_111_bar_5 \
   --lang-pairs zh_CN-en_XX,src_zh_CN_argument-tgt_en_XX_argument,src_zh_CN_cross-tgt_en_XX_cross \
   --encoder-normalize-before --decoder-normalize-before \
   --arch transformer --layernorm-embedding \
   --encoder-langtok "src" \
   --decoder-langtok \
   --task translation_multi_simple_epoch \
   --share-decoder-input-output-embed \
   --lang-dict $langs_txt \
   --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
   --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
   --lr-scheduler polynomial_decay --lr 5e-04 --warmup-updates 5000 --total-num-update 400000 \
   --dropout 0.2 --attention-dropout 0.1 --weight-decay 0.0 \
   --max-tokens 1536 --update-freq 24 \
   --save-interval 1 --save-interval-updates 2000 --keep-interval-updates 10 --no-epoch-checkpoints \
   --seed 222 --log-format simple --log-interval 200 \
   --skip-invalid-size-inputs-valid-test \
   --truncate-source --max-source-positions 1024 \
   --lang-tok-style 'mbart' \
   --ddp-backend legacy_ddp --sampling-method 'uniform' --use-embedding-CR true --fp16 --CR-embedding-scale 5 --share-mt-sum-tok true 
```

Train a multitask model with

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./CLS_CR/train_copy.py \
  data-bin/sum_multitask_1_1_1 \
  --save-dir models_trained/multilingual_zh2en_sa \
  --lang-pairs zh_CN-en_XX,src_zh_CN_cross-tgt_en_XX_cross \
  --encoder-normalize-before --decoder-normalize-before \
  --arch multilingual_transformer --layernorm-embedding \
  --task multilingual_translation \
  --encoder-langtok "src" \
  --decoder-langtok \
  --share-decoders \
  --share-encoders \
  --share-decoder-input-output-embed \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler polynomial_decay --lr 5e-04 --warmup-updates 5000 --total-num-update 400000 \
  --dropout 0.2 --attention-dropout 0.1 --weight-decay 0.0 \
  --max-tokens 2048 --update-freq 16 \
  --save-interval 1 --save-interval-updates 2000 --keep-interval-updates 10 --no-epoch-checkpoints \
  --seed 222 --log-format simple --log-interval 100 \
  --skip-invalid-size-inputs-valid-test \
  --truncate-source true --max-source-positions 1024 \
  --ddp-backend legacy_ddp --fp16
```
You can customize the multitask model (SA, SE, and SD) by adding or remove '--share-decoders' '--share-encoders' command.


Train a normal NCLS model with

```
langs_txt=lang.txt

# bart version
 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python ./CLS_CR/train_copy.py \
   data-bin/sum_multitask_1_1_1 \
   --save-dir models_trained/multilingual_zh2en_normal_new \
   --lang-pairs src_zh_CN_cross-tgt_en_XX_cross \
   --encoder-normalize-before --decoder-normalize-before \
   --arch transformer --layernorm-embedding \
   --encoder-langtok "src" \
   --decoder-langtok \
   --task translation_multi_simple_epoch \
   --share-decoder-input-output-embed \
   --lang-dict $langs_txt \
   --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
   --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
   --lr-scheduler polynomial_decay --lr 5e-04 --warmup-updates 5000 --total-num-update 200000 \
   --dropout 0.2 --attention-dropout 0.1 --weight-decay 0.0 \
   --max-tokens 2048 --update-freq 16 \
   --save-interval 1 --save-interval-updates 2000 --keep-interval-updates 10 --no-epoch-checkpoints \
   --seed 222 --log-format simple --log-interval 200 \
   --skip-invalid-size-inputs-valid-test \
   --truncate-source --max-source-positions 1024 \
   --lang-tok-style 'mbart' \
   --ddp-backend legacy_ddp --sampling-method 'uniform'  --fp16 --share-mt-sum-tok true

```

## Inference

Inference and evaluate a CSC model with a designated compression rate $\gamma$:

```



sentencepiece_model=/BART_PATH/mbart.cc25.v2
model_dir=models_trained/multilingual_zh2en_simple_111_cr_new  # fix if you moved the checkpoint
target_checkpoint=$model_dir/checkpoint_best.pt # fix if you moved the checkpoint
langs_txt=/home/ybai/projects/crcls/mbart.cc25.v2/lang.txt

for((i=1;i<=9;i++));

do
output_file=multi_task_sum_zh_en_0_$i
# fairseq-generate data-bin/wmt16_ro_en/processed \
CUDA_VISIBLE_DEVICES=0 python ./generate_copy.py data-bin/sum_multitask_1_1_1 \
  --path $target_checkpoint \
  --task  translation_multi_simple_epoch  \
  --encoder-langtok "src" \
  --decoder-langtok \
  --gen-subset test \
  --lang-dict $langs_txt \
  -t tgt_en_XX_cross -s src_zh_CN_cross \
  --bpe 'sentencepiece' --sentencepiece-model $sentencepiece_model/sentence.bpe.model \
  --skip-invalid-size-inputs-valid-test \
  --truncate-source --max-source-positions 1024 --no-repeat-ngram-size 3 \
  --max-len-b 150 \
  --sacrebleu \
   --lang-tok-style 'mbart' \
  --lang-pairs zh_CN-en_XX,src_zh_CN_monolingual-tgt_zh_CN_monolingual,src_zh_CN_cross-tgt_en_XX_cross \
  --batch-size 72 --inference-compression-rate 0.$i  > $model_dir/$output_file  # --remove-bpe 'sentencepiece'



echo 'compression rate = 0.'$i

cat $model_dir/$output_file | grep -P "^D" |sort -V |cut -f 3- | sed 's/\[ro_RO\]//g' > $model_dir/$output_file.hyp
cat $model_dir/$output_file | grep -P "^T" |sort -V |cut -f 2- | sed 's/\[ro_RO\]//g' > $model_dir/$output_file.ref

files2rouge $model_dir/$output_file.ref $model_dir/$output_file.hyp --ignore_empty_summary

# if the output langauge is Chinese
# python rouge_chinese.py -ref_path $model_dir/$output_file.ref -hyp_path $model_dir/$output_file.hyp
# files2rouge $model_dir/$output_file.ref0 $model_dir/$output_file.hyp0 --ignore_empty_summary

done
```
Inference and evaluate a CSC model with a oracle compression rate:

```
output_file=multi_task_sum_zh_en_oracle
# fairseq-generate data-bin/wmt16_ro_en/processed \
CUDA_VISIBLE_DEVICES=0 python ./generate_copy.py data-bin/sum_multitask_1_1_1 \
  --path $target_checkpoint \
  --task  translation_multi_simple_epoch  \
  --encoder-langtok "src" \
  --decoder-langtok \
  --gen-subset test \
  --lang-dict $langs_txt \
  -t tgt_en_XX_cross -s src_zh_CN_cross \
  --bpe 'sentencepiece' --sentencepiece-model $sentencepiece_model/sentence.bpe.model \
  --skip-invalid-size-inputs-valid-test \
  --truncate-source --max-source-positions 1024 --no-repeat-ngram-size 3 \
  --max-len-b 150 \
  --sacrebleu \
  --lang-tok-style 'mbart' \
  --lang-pairs zh_CN-en_XX,src_zh_CN_monolingual-tgt_zh_CN_monolingual \
  --batch-size 72  --inference-compression-rate 1.0 --oracle-compression-rate > $model_dir/$output_file  # --remove-bpe 'sentencepiece'

echo 'compression rate = oracle'
# 以下是能用命令的模版
cat $model_dir/$output_file | grep -P "^D" |sort -V |cut -f 3- | sed 's/\[ro_RO\]//g' > $model_dir/$output_file.hyp
cat $model_dir/$output_file | grep -P "^T" |sort -V |cut -f 2- | sed 's/\[ro_RO\]//g' > $model_dir/$output_file.ref

files2rouge $model_dir/$output_file.ref $model_dir/$output_file.hyp --ignore_empty_summary
```

<!-- 
# Citation

Please cite as:

``` bibtex


``` -->
