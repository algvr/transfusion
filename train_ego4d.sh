export CODE=$(pwd)
export DATA=$(pwd)/datasets
export RUNS=$(pwd)/runs
mkdir -p $CODE && echo "Set CODE to $CODE"
mkdir -p $DATA && echo "Set DATA to $DATA"
mkdir -p $RUNS && echo "Set RUNS to $RUNS"

cd runner
python run_experiment.py --config=nao/configs/ego_nao_res50_ego4d.yml "$@"