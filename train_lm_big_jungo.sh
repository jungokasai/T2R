gpu_id=$1
export CUDA_VISIBLE_DEVICES=${gpu_id}
update_freq=16

wd=0
drop=0.1
path=debug_lm
# nohup \
CUDA_VISIBLE_DEVICES=0-0,
python train.py \
--task language_modeling \
~/phillytools/data-bin/wikitext-103 \
--save-dir model/${path} \
--activation-fn "relu" \
--arch transformer_lm_wiki103 \
--decoder-layers 32 \
--decoder-attention-heads 8 \
--decoder-embed-dim 1024 \
--decoder-input-dim 1024 \
--decoder-output-dim 1024 \
--decoder-ffn-embed-dim 4096 \
--dropout 0.2 \
--adaptive-input \
--tie-adaptive-weights \
--adaptive-input-cutoff "20000,60000" \
--adaptive-softmax-cutoff "20000,60000" \
--adaptive-softmax-dropout 0.2 \
--attention-dropout 0.1 \
--activation-dropout 0.1 \
--decoder-layerdrop 0.2 \
--no-decoder-final-norm \
--tie-adaptive-proj \
--max-update 286000 \
--max-lr 1.0 \
--t-mult 2 \
--lr-period-updates 270000 \
--lr-scheduler cosine \
--lr-shrink 0.75 \
--warmup-updates 16000 \
--warmup-init-lr 1e-07 \
--min-lr 1e-09 \
--optimizer nag \
--lr 0.0001 \
--clip-norm 0.1 \
--criterion adaptive_loss \
--max-tokens 3072 \
--update-freq 16 \
--tokens-per-sample 512 \
--seed 1 \
--sample-break-mode none \
--skip-invalid-size-inputs-valid-test \
--ddp-backend=no_c10d \
--fp16 \
--fp16-init-scale 8 \
--train-subset valid \
--causal-proj-dim 64 \
--causal-tau 1.0 \
--reparam-proj \
--init-scale 6.0 \
--attn-only-training \
--random-feature mlp \
--restore-file ~/phillytools/fairseq/checkpoints/wiki103_ldrop0.2_dr0.2_win512_l32_1/checkpoint_last.pt \
--reset-optimizer \
--use-rfa \
#--cuda-causal-rfa \
#--restore-file ~/phillytools/fairseq/checkpoints/wiki103_ldrop0.2_dr0.2_win512_l32_1/checkpoint_last.pt \
#--reset-dataloader \
# > log/$path &
