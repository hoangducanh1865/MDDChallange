#!/bin/bash -l
#SBATCH --job-name=mdd_eval
#SBATCH --partition=defq
#SBATCH --output=/home/user14/anhhd/mdd/MDDChallange/logs/eval_%j.out
#SBATCH --error=/home/user14/anhhd/mdd/MDDChallange/logs/eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10g
#SBATCH --gres=gpu:a100:1

cd /home/user14/anhhd/mdd/MDDChallange

source /home/user14/miniconda3/bin/activate MDDChallange

mkdir -p logs

echo "Bắt đầu chạy eval..."
python main.py --mode eval
echo "Hoàn thành!"