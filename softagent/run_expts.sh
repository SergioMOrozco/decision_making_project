python dreamerv3/run_dreamer.py --env_name ClothFlatten --log_dir ./data/runs/variation_1/dreamer/clothflatten_seed100 --seed 100 --env_kwargs_num_variations 1 --train_epoch 100 --test_interval 1
python dreamerv3/run_dreamer.py --env_name RopeFlatten --log_dir ./data/runs/variation_1/dreamer/ropeflatten_seed100 --seed 100 --env_kwargs_num_variations 1 --train_epoch 100 --test_interval 1
python dreamerv3/run_dreamer.py --env_name ClothFold --log_dir ./data/runs/variation_1/dreamer/clothfold_seed100 --seed 100 --env_kwargs_num_variations 1 --train_epoch 100 --test_interval 1

python experiments/run_planet.py --env_name ClothFlatten --log_dir ./data/runs/variation_1/planet/clothflatten_seed100 --seed 100 --env_kwargs_num_variations 1 --train_epoch 100 --test_interval 1
python experiments/run_planet.py --env_name RopeFlatten --log_dir ./data/runs/variation_1/planet/ropeflatten_seed100 --seed 100 --env_kwargs_num_variations 1 --train_epoch 100 --test_interval 1
python experiments/run_planet.py --env_name ClothFold --log_dir ./data/runs/variation_1/planet/clothfold_seed100 --seed 100 --env_kwargs_num_variations 1 --train_epoch 100 --test_interval 1

#python experiments/run_planet.py --env_name ClothFlatten --log_dir ./data/clothflatten_seed110 --seed 110 --env_kwargs_num_variations 100 --train_epoch 10 --test_interval 1
#python experiments/run_planet.py --env_name RopeFlatten --log_dir ./data/ropeflatten_seed110 --seed 110 --env_kwargs_num_variations 100 --train_epoch 10 --test_interval 1
#python experiments/run_planet.py --env_name ClothFold --log_dir ./data/clothfold_seed110 --seed 110 --env_kwargs_num_variations 100 --train_epoch 10 --test_interval 1
