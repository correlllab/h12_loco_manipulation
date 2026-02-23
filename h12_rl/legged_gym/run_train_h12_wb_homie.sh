#!/usr/bin/env bash
# Run training for h12_wb_homie using THIS repo's legged_gym (not unitree_rl_gym).
#
# Requires the HIM fork of rsl_rl (HIMOnPolicyRunner, etc.). Set HIM_RSL_RL_DIR to the
# directory that contains that rsl_rl package (e.g. unitree_rl_gym) so it is used instead
# of the standard rsl_rl:
#   export HIM_RSL_RL_DIR=/home/niraj/gym_projects/unitree_rl_gym
#   ./run_train_h12_wb_homie.sh
#
# Usage: from h12_rl/legged_gym run:  ./run_train_h12_wb_homie.sh  [extra args...]
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# This repo's legged_gym must come first so h12_wb_homie is registered; then HIM rsl_rl if set
if [ -n "${HIM_RSL_RL_DIR:-}" ]; then
  export PYTHONPATH="$SCRIPT_DIR:$HIM_RSL_RL_DIR:$PYTHONPATH"
else
  export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
fi
python legged_gym/scripts/train.py --task h12_wb_homie --num_envs 4096 --headless --max_iterations 10000 "$@"
