#!/system/bin/sh
# Best-effort Exynos/Android sysfs sampler.
# Logs monotonic_ns plus whatever devfreq/cpufreq/thermal nodes are present.
# Usage:
#   sh collect_exynos_sysfs_metrics.sh /data/local/tmp/exynos_metrics.csv 20
#   arg1: output csv path (default /data/local/tmp/exynos_metrics.csv)
#   arg2: sampling period in ms (default 20)

OUT="${1:-/data/local/tmp/exynos_metrics.csv}"
PERIOD_MS="${2:-20}"

now_ns() {
  awk '{printf "%.0f\n", $1*1000000000}' /proc/uptime
}

append_if_exists() {
  KEY="$1"
  PATHNAME="$2"
  if [ -f "$PATHNAME" ]; then
    VAL=$(cat "$PATHNAME" 2>/dev/null | tr -d '\n' | tr ',' ';')
    printf ",%s" "$VAL"
  else
    printf ","
  fi
}

DEVFREQ_DIRS=$(find /sys/class/devfreq -maxdepth 1 -mindepth 1 -type d 2>/dev/null)
CPUFREQ_DIRS=$(find /sys/devices/system/cpu/cpufreq -maxdepth 1 -mindepth 1 -type d -name 'policy*' 2>/dev/null)
THERMAL_DIRS=$(find /sys/class/thermal -maxdepth 1 -mindepth 1 -type d -name 'thermal_zone*' 2>/dev/null)

# header
{
  printf "monotonic_ns"
  for d in $DEVFREQ_DIRS; do
    base=$(basename "$d")
    printf ",devfreq_%s_name,devfreq_%s_cur_freq,devfreq_%s_min_freq,devfreq_%s_max_freq,devfreq_%s_load,devfreq_%s_busy_time,devfreq_%s_total_time" "$base" "$base" "$base" "$base" "$base" "$base" "$base"
  done
  for d in $CPUFREQ_DIRS; do
    base=$(basename "$d")
    printf ",cpufreq_%s_cur,cpufreq_%s_min,cpufreq_%s_max" "$base" "$base" "$base"
  done
  for d in $THERMAL_DIRS; do
    base=$(basename "$d")
    printf ",thermal_%s_type,thermal_%s_temp" "$base" "$base"
  done
  printf "\n"
} > "$OUT"

while true; do
  {
    printf "%s" "$(now_ns)"

    for d in $DEVFREQ_DIRS; do
      base=$(basename "$d")
      append_if_exists "name" "$d/name"
      append_if_exists "cur_freq" "$d/cur_freq"
      append_if_exists "min_freq" "$d/min_freq"
      append_if_exists "max_freq" "$d/max_freq"
      append_if_exists "load" "$d/load"
      append_if_exists "busy_time" "$d/busy_time"
      append_if_exists "total_time" "$d/total_time"
    done

    for d in $CPUFREQ_DIRS; do
      append_if_exists "scaling_cur_freq" "$d/scaling_cur_freq"
      append_if_exists "scaling_min_freq" "$d/scaling_min_freq"
      append_if_exists "scaling_max_freq" "$d/scaling_max_freq"
    done

    for d in $THERMAL_DIRS; do
      append_if_exists "type" "$d/type"
      append_if_exists "temp" "$d/temp"
    done
    printf "\n"
  } >> "$OUT"

  usleep $((PERIOD_MS * 1000))
done
