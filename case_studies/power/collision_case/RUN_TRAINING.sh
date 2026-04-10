#!/bin/bash
# HERON + PowerGym collision training (three MG policies)

echo "Run from: PowerGym-main/case_studies/power"
echo ""
echo "Shared reward example:"
echo '  nohup python -m collision_case.train_collision --algo=PPO --num-agents=3 --share-reward \'
echo '    --log-path=collision_case/collision_shared_300.csv --stop-iters=300 \'
echo '    > collision_case/train_shared_300.log 2>&1 &'
echo ""
echo "Independent reward: omit --share-reward"
echo "Async schedules: add --enable-async"
echo ""
echo "Monitor: tail -f collision_case/train_shared_300.log"
echo "Process: ps aux | grep train_collision"
