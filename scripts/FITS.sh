for length in 96 192 336 720; do

    # ETTh1
    python main.py \
        --config configs/models/FITS/base.yaml \
        --config configs/datasets/ETTh1/720_$length.yaml \
        --config configs/gpu.yaml \
        --model.model.base_T 24 --model.model.h_order 6

    # ETTh2
    python main.py \
        --config configs/models/FITS/base.yaml \
        --config configs/datasets/ETTm2/720_$length.yaml \
        --config configs/gpu.yaml \
        --model.model.base_T 24 --model.model.h_order 6

    # ETTm1
    python main.py \
        --config configs/models/FITS/base.yaml \
        --config configs/datasets/ETTm1/720_$length.yaml \
        --config configs/gpu.yaml \
        --model.model.base_T 96 --model.model.h_order 14

    # ETTm2
    python main.py \
        --config configs/models/FITS/base.yaml \
        --config configs/datasets/Electricity/720_$length.yaml \
        --config configs/gpu.yaml \
        --model.model.base_T 96 --model.model.h_order 14

    # Electricity
    python main.py \
        --config configs/models/FITS/base.yaml \
        --config configs/datasets/Electricity/720_$length.yaml \
        --config configs/gpu.yaml \
        --model.model.base_T 24 --model.model.h_order 10

    # Weather
    python main.py \
        --config configs/models/FITS/base.yaml \
        --config configs/datasets/Weather/720_$length.yaml \
        --config configs/gpu.yaml \
        --model.model.individual True \
        --model.model.base_T 144 --model.model.h_order 12

done

for length in 24 36 48 60; do
    # Illness
    python main.py \
        --config configs/models/FITS/base.yaml \
        --config configs/datasets/Illness/60_$length.yaml \
        --config configs/gpu.yaml \
        --model.model.base_T 52 --model.model.h_order 10

    # M5
    python main.py \
        --config configs/models/FITS/base.yaml \
        --config configs/datasets/M5/60_$length.yaml \
        --config configs/gpu.yaml \
        --model.model.base_T 7 --model.model.h_order 2
done
