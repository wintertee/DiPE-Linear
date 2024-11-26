for length in 96 192 336 720; do

    # ETTh1
    python main.py \
        --config configs/models/DLinear/base.yaml \
        --config configs/datasets/ETTh1/720_$length.yaml \
        --config configs/gpu.yaml

    # ETTh2
    python main.py \
        --config configs/models/DLinear/base.yaml \
        --config configs/datasets/ETTm2/720_$length.yaml \
        --config configs/gpu.yaml

    # ETTm1
    python main.py \
        --config configs/models/DLinear/base.yaml \
        --config configs/datasets/ETTm1/720_$length.yaml \
        --config configs/gpu.yaml

    # ETTm2
    python main.py \
        --config configs/models/DLinear/base.yaml \
        --config configs/datasets/Electricity/720_$length.yaml \
        --config configs/gpu.yaml

    # Electricity
    python main.py \
        --config configs/models/DLinear/base.yaml \
        --config configs/datasets/Electricity/720_$length.yaml \
        --config configs/gpu.yaml

    # Weather
    python main.py \
        --config configs/models/DLinear/base.yaml \
        --config configs/datasets/Weather/720_$length.yaml \
        --config configs/gpu.yaml \
        --model.model.individual True

done

for length in 24 36 48 60; do
    # Illness
    python main.py \
        --config configs/models/DLinear/base.yaml \
        --config configs/datasets/Illness/60_$length.yaml \
        --config configs/gpu.yaml

    # M5
    python main.py \
        --config configs/models/DLinear/base.yaml \
        --config configs/datasets/M5/60_$length.yaml \
        --config configs/gpu.yaml
done
