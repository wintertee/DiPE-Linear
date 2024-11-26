for length in 96 192 336 720; do

    # ETTh1
    python main.py \
        --config configs/models/DiPE/base.yaml \
        --config configs/datasets/ETTh1/720_$length.yaml \
        --config configs/gpu.yaml \
        --model.model.num_experts 1 \
        --model.model.loss_alpha 1 \
        --model.model.use_revin True

    # ETTh2
    python main.py \
        --config configs/models/DiPE/base.yaml \
        --config configs/datasets/ETTm2/720_$length.yaml \
        --config configs/gpu.yaml \
        --model.model.num_experts 3 \
        --model.model.loss_alpha 0.9 \
        --model.model.use_revin True

    # ETTm1
    python main.py \
        --config configs/models/DiPE/base.yaml \
        --config configs/datasets/ETTm1/720_$length.yaml \
        --config configs/gpu.yaml \
        --model.model.num_experts 1 \
        --model.model.loss_alpha 1 \
        --model.model.use_revin True

    # ETTm2
    python main.py \
        --config configs/models/DiPE/base.yaml \
        --config configs/datasets/Electricity/720_$length.yaml \
        --config configs/gpu.yaml \
        --model.model.num_experts 1 \
        --model.model.loss_alpha 0.9 \
        --model.model.use_revin True

    # Electricity
    python main.py \
        --config configs/models/DiPE/base.yaml \
        --config configs/datasets/Electricity/720_$length.yaml \
        --config configs/gpu.yaml \
        --model.model.num_experts 4 \
        --model.model.loss_alpha 0.3 \
        --model.model.use_revin False

    # Weather
    python main.py \
        --config configs/models/DiPE/base.yaml \
        --config configs/datasets/Weather/720_$length.yaml \
        --config configs/gpu.yaml \
        --model.model.num_experts 4 \
        --model.model.loss_alpha 0.9 \
        --model.model.use_revin False

done

for length in 24 36 48 60; do
    # Illness
    python main.py \
        --config configs/models/DiPE/base.yaml \
        --config configs/datasets/Illness/60_$length.yaml \
        --config configs/gpu.yaml \
        --model.model.num_experts 1 \
        --model.model.loss_alpha 1 \
        --model.model.use_revin True

    # M5
    python main.py \
        --config configs/models/DiPE/base.yaml \
        --config configs/datasets/M5/60_$length.yaml \
        --config configs/gpu.yaml \
        --model.model.num_experts 4 \
        --model.model.loss_alpha 0 \
        --model.model.use_revin True
done
