#!/usr/bin/env bash
# ===========================================================================
#  run_benchmark.sh
#
#  Reproduces every benchmark defined in utils/model_configuration.py
#  (experiment_settings) by calling evaluate_nonstgym_parallel.py once per
#  experiment key.
#
#  - Seeds are fixed inside evaluate_nonstgym_parallel.py
#    (base_seed = 2721413865, seed of run k = base_seed + k).
#  - Drift is controlled by the "drift_disabled" flag in each config, so
#    --nodrift is intentionally NOT passed. (Note: --nodrift is parsed as a
#    string, so even "--nodrift False" would disable drift.)
#  - Results go to ./results/*.json, one console log per experiment goes
#    to ./logs/<experiment>.log.
#
#  Usage:
#    chmod +x run_benchmark.sh
#    ./run_benchmark.sh
#
#  PYTHON and CPU_JOBS can be overridden from the environment, e.g.
#    PYTHON=.venv/bin/python CPU_JOBS=8 ./run_benchmark.sh
# ===========================================================================

set -u

# Run from the directory this script lives in (equivalent of cd /d "%~dp0")
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" || exit 1

# ---- User settings --------------------------------------------------------
PYTHON="${PYTHON:-python3}"
SCRIPT="src/evaluate_nonstgym_parallel.py"
# Worker processes for MLP (CPU) experiments
CPU_JOBS="${CPU_JOBS:-20}"
# ---------------------------------------------------------------------------

LOG_DIR="logs"
mkdir -p "$LOG_DIR" results

FAILED=0
FAILED_LIST=""

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

# ---------------------------------------------------------------------------
#  run <experiment_key> <device> <n_jobs>
# ---------------------------------------------------------------------------
run() {
  local exp="$1" dev="$2" jobs="$3"
  echo
  echo "[$(timestamp)] START  $exp  (device=$dev, n-jobs=$jobs)"
  echo "  $PYTHON $SCRIPT --env $exp --n-jobs $jobs --device $dev"
  if "$PYTHON" "$SCRIPT" --env "$exp" --n-jobs "$jobs" --device "$dev" \
       > "$LOG_DIR/$exp.log" 2>&1; then
    echo "[$(timestamp)] DONE   $exp"
  else
    echo "[$(timestamp)] ERROR  $exp  - see $LOG_DIR/$exp.log"
    FAILED=$((FAILED + 1))
    FAILED_LIST="$FAILED_LIST $exp"
  fi
}


# ---- Classic control (DQN / PPO) ----
run CartPole                     cpu  "$CPU_JOBS"
run MountainCar                  cpu  "$CPU_JOBS"

# ---- Classic control (SAC / PPO), drift disabled in config ----
run MountainCarContinuousNoSDE   cpu  "$CPU_JOBS"

# ---- Lunar Lander ----
run MultiPadLLDiscrete           cpu  "$CPU_JOBS"
run MultiPadLLContinuous         cpu  "$CPU_JOBS"

# ---- MuJoCo (SAC / PPO) ----
run InvertedPendulum             cpu  "$CPU_JOBS"
run Ant                          cpu  "$CPU_JOBS"
run AntNewSampling               cpu  "$CPU_JOBS"

# ---- ChooseBox ----
run ChooseBox                    cpu  "$CPU_JOBS"
run ChooseBoxLimited             cpu  "$CPU_JOBS"

# ---- Exploration ----
run RiverSwim                    cpu  "$CPU_JOBS"

# ===========================================================================
#  No-drift controls (drift_disabled = True in config)
# ===========================================================================
run CartPoleNoDrift              cpu  "$CPU_JOBS"
run MountainCarNoDrift           cpu  "$CPU_JOBS"
run MultiPadLLDiscreteNoDrift    cpu  "$CPU_JOBS"
run MultiPadLLContinuousNoDrift  cpu  "$CPU_JOBS"
run InvertedPendulumNoDrift      cpu  "$CPU_JOBS"
run AntNoDrift                   cpu  "$CPU_JOBS"
run ChooseBoxNoDrift             cpu  "$CPU_JOBS"


# ===========================================================================
echo
echo "==========================================================================="
echo "All experiments finished: $(timestamp)"
if [ "$FAILED" -gt 0 ]; then
  echo "$FAILED experiment(s) exited with an error:$FAILED_LIST"
  echo "See the corresponding files in $LOG_DIR/"
else
  echo "No experiment crashed. Check the logs for \"[FAILED]\" lines, which mark"
  echo "individual seeds that errored but did not stop the sweep."
fi
echo "==========================================================================="
exit "$FAILED"
