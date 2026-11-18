#!/bin/sh
# hicache_pool.sh - HiCache L3 缓存池（KUNPENG_HICACHE_L3_DIR）管理脚本。
#
# 纯 sh + GNU find/awk，无第三方依赖。
# 任何删除动作默认只预览（dry-run），必须加 -f 才真正执行。
#
# 目录结构（HiCacheFile 后端，一页一文件）：
#   {hash}{config_suffix}.bin           KV 页 blob（target 各层 + MTP draft 层）
#   {hash}@idxk{config_suffix}.bin      index-K 页 blob（target + draft）

set -u

DIR="${HICACHE_DIR:-${KUNPENG_HICACHE_L3_DIR:-${SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR:-}}}"
DAYS=0
MAX_GB=0
INTERVAL=600
LINES=50
JOBS=32
FORCE=0
FILTER=""
CLEAN_DATE=""
DATE_BEFORE=0

usage() {
cat <<'EOF'
用法:
  sh hicache_pool.sh <命令> [数值参数] [选项]

命令:
  info                查看缓存池概览：总量、页文件数量、按天分布
  clean [天数]        删除 N 天前写入的页（需指定天数或 --date）
  size  [上限GB]      超过容量上限时，从最久未使用开始删（需指定上限）
  orphan              删除孤儿文件（有 d:/@idxk 但缺少对应 target 文件）
  watch-once          执行一轮清理（clean + size），适合挂 cron
  start               启动后台守护进程，周期性执行 watch-once
  stop                停止后台守护进程
  status              查看守护进程状态与最近日志
  log                 查看守护日志
  -h, --help          显示本帮助

选项:
  --dir <目录>        缓存池目录（默认取 KUNPENG_HICACHE_L3_DIR 或
                      SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR）
  --days <N>          clean/watch 的过期天数（滚动 N×24 小时，非自然日；
                      默认 0，即不按天清理；>0 才启用）
  --date <YYYY-MM-DD> clean 只删除该自然日写入的页（优先于 --days）
  --before            配合 --date：删除该日期及之前的所有页
  --max-gb <G>        size/watch 的容量上限 GB（默认 0，即不做容量淘汰；
                      使用 size 或守护时必须显式指定）
  -i, --interval <秒>  守护进程检查间隔，仅 start 使用（默认 600，最小 60）
  -f, --force         真正执行删除
  -j, --jobs <N>      删除/校验并发数（默认 32，1 为串行）
  -n, --lines <N>     log 显示的行数（默认 50）
  --model <子串>      只处理文件名中包含该子串的条目（多模型共用目录时用，
                      例如 --model GLM-5.3）
  -h, --help          显示本帮助

清理规则:
  1) clean：按 mtime 删除过期页；需指定 --days 或 --date，未指定时不删除；
  2) size：超过 --max-gb 时，按 max(atime, mtime) 从最久未使用开始逐页删除；
     文件系统不更新 atime 时（如 dtfs）等价于按写入时间淘汰，脚本会提示；
  3) --days 与 --max-gb 默认都是 0，即默认不做任何清理；
  4) 默认只预览，加 -f 执行删除。

后台运行方式（二选一）:

  一、后台常驻
    sh hicache_pool.sh start --days 7 --max-gb 500 -i 600 -f
    -i 为轮询间隔（秒）；status 查看状态，stop 停止，log 查看日志。

  二、cron 定时任务
    执行 crontab -e，写入下面一行并保存，cron 会自动在后台周期执行:
    */10 * * * * KUNPENG_HICACHE_L3_DIR=/your/pool sh /path/hicache_pool.sh watch-once --days 7 --max-gb 500 -f >> /path/hicache_pool.log 2>&1

    该行由“时间规则 + 执行命令”组成：开头的 */10 * * * * 是时间规则
    （字段依次为 分 时 日 月 周），表示每 10 分钟触发一次；其余部分是
    cron 到点执行的命令。
    cron 不继承 shell 环境变量，需用 --dir 或前置 KUNPENG_HICACHE_L3_DIR=...；
    不加 -f 只预览；建议重定向日志，避免 cron 发邮件。

  两种方式任选其一，不要同时启用。

示例:
  # 查看概览与按天分布
  sh hicache_pool.sh info

  # 预览：删除 7 天前的页
  sh hicache_pool.sh clean 7

  # 真正删除 7 天前的页
  sh hicache_pool.sh clean 7 -f

  # 预览：容量控制在 500GB 以内
  sh hicache_pool.sh size 500

  # 只删除某一天写入的页
  sh hicache_pool.sh clean --date 2026-11-12 -f

  # 删除该日期及之前的所有页
  sh hicache_pool.sh clean --date 2026-11-12 --before -f

  # 清理孤儿文件
  sh hicache_pool.sh orphan -f

  # 后台守护：每 10 分钟检查一次，>7 天或 >500GB 就清理
  sh hicache_pool.sh start --days 7 --max-gb 500 -i 600 -f

  # 只按容量淘汰（默认 --days 0，不做按天清理）
  sh hicache_pool.sh start --max-gb 500 -i 600 -f

  # 多模型共用目录时，只处理 GLM-5.3 的缓存
  sh hicache_pool.sh start --model GLM-5.3 -f
EOF
}

die() {
    echo "错误: $*" >&2
    exit 2
}

need_dir() {
    [ -n "$DIR" ] || die "未指定缓存目录，请用 --dir 或设置 KUNPENG_HICACHE_L3_DIR"
    [ -d "$DIR" ] || die "目录不存在: $DIR"
}

human_gb() {
    awk -v b="$1" 'BEGIN {
        if (b !~ /^[0-9]+$/) b = 0
        printf "%.2f GB", b / 1073741824
    }'
}

# 本地时区偏移（秒），用于把 mtime 的 UTC 日桶换算成本地日期。
local_tz_offset() {
    ltz=$(date +%z)
    lsign=$(printf '%s' "$ltz" | cut -c1)
    lhh=$(printf '%s' "$ltz" | cut -c2-3 | sed 's/^0*//')
    lmm=$(printf '%s' "$ltz" | cut -c4-5 | sed 's/^0*//')
    [ -z "$lhh" ] && lhh=0
    [ -z "$lmm" ] && lmm=0
    loff=$((lhh * 3600 + lmm * 60))
    [ "$lsign" = "-" ] && loff=$((-loff))
    printf '%s\n' "$loff"
}

# 扫描目录：readdir 只取文件名（不 stat），size/atime/mtime 交给并发 stat。
# 每个并发进程写自己的临时文件，避免多进程同时写管道把行截断。
# 输出每行：size atime mtime name
scan_meta() {
    sm_tmpd=$(mktemp -d) || return 1
    find "$DIR" -maxdepth 1 -name '*.bin' -printf '%f\n' 2>/dev/null |
    xargs -r -P "$JOBS" -n 256 sh -c '
        d=$1; t=$2; shift 2
        ( cd "$d" && stat -c "%s %X %Y %n" -- "$@" ) > "$t/$$" 2>/dev/null
    ' sh "$DIR" "$sm_tmpd"
    cat "$sm_tmpd"/* 2>/dev/null
    rm -rf "$sm_tmpd"
}

cmd_info() {
    need_dir
    t0=$(date +%s)
    echo "缓存目录: $DIR"
    daysf=$(mktemp) || die "mktemp 失败"
    totf=$(mktemp) || die "mktemp 失败"
    # 本地时区偏移（秒），把 mtime 的 UTC 日桶换算成本地日期。
    off=$(local_tz_offset)

    scan_meta | awk -v flt="$FILTER" -v daysf="$daysf" -v totf="$totf" -v off="$off" '
    {
        f = $4
        if (flt != "" && index(f, flt) == 0) next
        bytes += $1
        total++
        d = int(($3 + off) / 86400)
        day[d] += $1
        dayn[d]++
        is_d = (substr(f, 1, 2) == "d:")
        is_i = (index(f, "@idxk") > 0)
        if (is_d && is_i)      d_idx++
        else if (is_d)         d_kv++
        else if (is_i)         t_idx++
        else                   t_kv++
    }
    END {
        printf "%d %d\n", total + 0, bytes + 0 > totf
        printf "%d %d %d %d\n", t_kv + 0, t_idx + 0, d_kv + 0, d_idx + 0 > (daysf ".t")
        for (d in day) printf "%d %d %.2f\n", d, dayn[d], day[d] / 1073741824 >> daysf
    }'
    read total bytes < "$totf"
    read t_kv t_idx d_kv d_idx < "$daysf.t"
    echo "文件总数: $total"
    echo "占用空间: $bytes 字节 ($(human_gb "$bytes"))"
    echo ""
    echo "按文件类型:"
    printf "  KV 页 blob      : %d 页\n" "$t_kv"
    printf "  index-K 页 blob : %d 页\n" "$t_idx"
    echo ""
    echo "按写入日期（mtime）:"
    sort -n "$daysf" | while read -r d n g; do
        ds=$(date -d "@$((d * 86400 - off))" +%F)
        printf "  %s  %8d 页  %8.2f GB\n" "$ds" "$n" "$g"
    done
    rm -f "$daysf" "$daysf.t" "$totf"
    echo ""
    echo "耗时: $(( $(date +%s) - t0 )) 秒"
}

cmd_clean() {
    need_dir
    t0=$(date +%s)
    now=$(date +%s)
    if [ -n "$CLEAN_DATE" ]; then
        day_start=$(date -d "$CLEAN_DATE 00:00:00" +%s 2>/dev/null) || die "无效日期: $CLEAN_DATE"
        [ "$(date -d "@$day_start" +%F 2>/dev/null)" = "$CLEAN_DATE" ] || die "日期格式应为 YYYY-MM-DD: $CLEAN_DATE"
        end=$((day_start + 86400))
        if [ "$DATE_BEFORE" = 1 ]; then
            start=0
            label="$CLEAN_DATE 及之前的文件"
            range_msg="$CLEAN_DATE 及之前（< $(date -d "@$end" '+%F %T')）"
        else
            start=$day_start
            label="$CLEAN_DATE 的文件"
            range_msg="$(date -d "@$start" '+%F %T') ~ $(date -d "@$end" '+%F %T')"
        fi
    else
        if [ "$DAYS" -le 0 ] 2>/dev/null; then
            echo "未指定过期天数（--days 0），跳过按天清理"
            return 0
        fi
        start=0
        end=$((now - DAYS * 86400))
        label="超过 $DAYS 天的文件"
        range_msg="mtime < $(date -d "@$end" '+%F %T')（滚动 $DAYS×24 小时）"
    fi
    off=$(local_tz_offset)
    statf=$(mktemp) || die "mktemp 失败"
    dayf=$(mktemp) || die "mktemp 失败"

    # 一次 readdir + 并发 stat；按 mtime 筛选后交给 xargs 并发删除，
    # 同时按自然日汇总条数/大小，供预览和删除前确认。
    if [ "$FORCE" = 1 ]; then
        scan_meta |
        awk -v start="$start" -v end="$end" -v flt="$FILTER" -v statf="$statf" -v dayf="$dayf" -v off="$off" '
        {
            if ($3 < start) next
            if (end > 0 && $3 >= end) next
            f = $4
            if (flt != "" && index(f, flt) == 0) next
            print f
            n++
            s += $1
            d = int(($3 + off) / 86400)
            day[d] += $1
            dayn[d]++
        }
        END {
            printf "%d %d\n", n + 0, s + 0 > statf
            for (d in day) printf "%d %d %.2f\n", d, dayn[d], day[d] / 1073741824 >> dayf
        }' |
        ( cd "$DIR" && xargs -r -P "$JOBS" -n 64 rm -f -- )
    else
        scan_meta |
        awk -v start="$start" -v end="$end" -v flt="$FILTER" -v statf="$statf" -v dayf="$dayf" -v off="$off" '
        {
            if ($3 < start) next
            if (end > 0 && $3 >= end) next
            f = $4
            if (flt != "" && index(f, flt) == 0) next
            n++
            s += $1
            d = int(($3 + off) / 86400)
            day[d] += $1
            dayn[d]++
        }
        END {
            printf "%d %d\n", n + 0, s + 0 > statf
            for (d in day) printf "%d %d %.2f\n", d, dayn[d], day[d] / 1073741824 >> dayf
        }'
    fi
    read cnt total < "$statf"
    echo "目录: $DIR"
    echo "范围: $range_msg"
    if [ -s "$dayf" ]; then
        sort -n "$dayf" | while read -r d n g; do
            ds=$(date -d "@$((d * 86400 - off))" +%F)
            printf "  %s  %8d 页  %8.2f GB\n" "$ds" "$n" "$g"
        done
    fi
    echo "$label: $cnt 个"
    if [ "$FORCE" = 1 ]; then
        echo "本次删除 $cnt 个文件，释放 $(human_gb "$total")"
    else
        echo "本次预计删除 $cnt 个文件，释放 $(human_gb "$total")"
    fi
    rm -f "$statf" "$dayf"
    echo "耗时: $(( $(date +%s) - t0 )) 秒"
}

cmd_orphan() {
    need_dir
    t0=$(date +%s)
    orphans=$(mktemp) || die "mktemp 失败"

    # draft 文件：去掉 d: 前缀后对应的 target 是否存在。按 64 个一批并发检查。
    find "$DIR" -maxdepth 1 -type f -name 'd:*' -printf '%f\n' 2>/dev/null |
    xargs -r -P "$JOBS" -n 64 sh -c '
        d=$1; flt=$2; shift 2
        for f in "$@"; do
            if [ -n "$flt" ]; then
                case "$f" in *"$flt"*) ;; *) continue ;; esac
            fi
            [ -e "$d/${f#d:}" ] || printf "%s\n" "$f"
        done
    ' sh "$DIR" "$FILTER" >> "$orphans"

    # index-K 文件：去掉 @idxk 后对应的 target KV 是否存在。
    find "$DIR" -maxdepth 1 -type f -name '*@idxk*' -printf '%f\n' 2>/dev/null |
    xargs -r -P "$JOBS" -n 64 sh -c '
        d=$1; flt=$2; shift 2
        for f in "$@"; do
            if [ -n "$flt" ]; then
                case "$f" in *"$flt"*) ;; *) continue ;; esac
            fi
            kv="${f%%@idxk*}${f#*@idxk}"
            [ -e "$d/${kv#d:}" ] || printf "%s\n" "$f"
        done
    ' sh "$DIR" "$FILTER" >> "$orphans"

    cnt=$(wc -l < "$orphans" | tr -d ' ')
    echo "孤儿文件: $cnt 个"
    if [ "$cnt" -gt 0 ]; then
        if [ "$FORCE" = 1 ]; then
            ( cd "$DIR" && xargs -r -P "$JOBS" -n 64 rm -f -- < "$orphans" )
            echo "已删除 $cnt 个孤儿文件"
        else
            head -n 20 "$orphans" | sed 's/^/  [预览] /'
            [ "$cnt" -gt 20 ] && echo "  ..."
        fi
    fi
    rm -f "$orphans"
    echo "耗时: $(( $(date +%s) - t0 )) 秒"
}

cmd_size() {
    need_dir
    [ "$MAX_GB" -gt 0 ] 2>/dev/null || die "请指定容量上限：size <GB> 或 --max-gb <GB>"
    t0=$(date +%s)
    limit=$((MAX_GB * 1073741824))
    listing=$(mktemp) || die "mktemp 失败"
    scan_meta > "$listing"
    total=$(awk '{ s += $1 } END { print s + 0 }' "$listing")
    echo "目录: $DIR"
    echo "当前占用: $(human_gb "$total")，上限: ${MAX_GB} GB"
    if [ "$total" -le "$limit" ]; then
        rm -f "$listing"
        echo "未超上限，无需清理"
        return 0
    fi
    need=$((total - limit))
    echo "需释放: $(human_gb "$need")"
    # （dtfs/nfs 常见），此时 max(atime, mtime) 就等于 mtime。
    if [ "$(awk '$2 > $3 { n++ } END { print n + 0 }' "$listing")" -eq 0 ]; then
        echo "提示: 文件系统未更新 atime，等价于按写入时间(mtime)淘汰"
    fi

    now=$(date +%s)
    guard=$((10 * 60))
    dellist=$(mktemp) || die "mktemp 失败"
    tbl=$(mktemp) || die "mktemp 失败"
    summary=$(awk -v now="$now" -v guard="$guard" '
    {
        k = ($2 > $3) ? $2 : $3
        if (now - k < guard) next
        printf "%s %s %s\n", k, $1, $4
    }' "$listing" | sort -k1,1n --parallel="$JOBS" |
    awk -v need="$need" -v flt="$FILTER" -v dl="$dellist" -v tb="$tbl" '
    BEGIN { PROCINFO["sorted_in"] = "@ind_str_asc" }
    {
        name = $3
        if (flt != "" && index(name, flt) == 0) next
        if (freed >= need) next
        print name > dl
        freed += $2
        cnt++
        d = strftime("%F", $1)
        dc[d]++
        ds[d] += $2
    }
    END {
        for (d in dc) {
            printf "  %s %8d 个 %8.2f GB\n", d, dc[d],
                   ds[d] / 1073741824 > tb
        }
        printf "%d %d\n", cnt + 0, freed + 0
    }')
    rm -f "$listing"
    cnt=${summary%% *}
    freed=${summary##* }
    if [ "$cnt" -gt 0 ]; then
        echo "按 max(atime, mtime) 最旧优先，选中:"
        cat "$tbl"
    fi
    rm -f "$tbl"
    if [ "$FORCE" = 1 ]; then
        ( cd "$DIR" && xargs -r -P "$JOBS" -n 64 rm -f -- < "$dellist" )
        echo "本次删除 $cnt 个文件，释放 $(human_gb "$freed")"
    else
        echo "本次预计删除 $cnt 个文件，释放 $(human_gb "$freed")"
    fi
    rm -f "$dellist"
    echo "清理结束，耗时 $(( $(date +%s) - t0 )) 秒"
}

cmd_watch_once() {
    if [ "$DAYS" -le 0 ] 2>/dev/null && [ "$MAX_GB" -le 0 ] 2>/dev/null; then
        echo "未指定清理策略（--days / --max-gb），跳过本轮"
        return 0
    fi
    [ "$DAYS" -gt 0 ] 2>/dev/null && cmd_clean
    [ "$MAX_GB" -gt 0 ] 2>/dev/null && cmd_size
    return 0
}

cmd_daemon() {
    need_dir
    adm="$DIR/.hicache_admin"
    mkdir -p "$adm"
    logf="$adm/daemon.log"
    echo "===== 守护启动 $(date '+%Y-%m-%d %H:%M:%S') dir=$DIR days=$DAYS max_gb=$MAX_GB interval=${INTERVAL}s force=$FORCE =====" >> "$logf"
    while :; do
        {
            echo "----- $(date '+%Y-%m-%d %H:%M:%S') 开始一轮 -----"
            cmd_watch_once
            echo "----- $(date '+%Y-%m-%d %H:%M:%S') 本轮结束 -----"
        } >> "$logf" 2>&1
        if [ -f "$logf" ] && [ "$(wc -c < "$logf")" -gt 10485760 ]; then
            tail -n 1000 "$logf" > "$logf.tmp" && mv "$logf.tmp" "$logf"
        fi
        sleep "$INTERVAL"
    done
}

cmd_start() {
    need_dir
    if [ "$DAYS" -le 0 ] 2>/dev/null && [ "$MAX_GB" -le 0 ] 2>/dev/null; then
        die "请至少指定一种清理策略：--days <N> 或 --max-gb <G>"
    fi
    adm="$DIR/.hicache_admin"
    mkdir -p "$adm"
    pidf="$adm/daemon.pid"
    logf="$adm/daemon.log"
    if [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
        echo "守护进程已在运行 (pid=$(cat "$pidf"))"
        return 0
    fi
    if [ "$FORCE" = 1 ]; then
        if [ -n "$FILTER" ]; then
            nohup sh "$0" __daemon --dir "$DIR" --days "$DAYS" --max-gb "$MAX_GB" -i "$INTERVAL" --model "$FILTER" -f >> "$logf" 2>&1 &
        else
            nohup sh "$0" __daemon --dir "$DIR" --days "$DAYS" --max-gb "$MAX_GB" -i "$INTERVAL" -f >> "$logf" 2>&1 &
        fi
    else
        if [ -n "$FILTER" ]; then
            nohup sh "$0" __daemon --dir "$DIR" --days "$DAYS" --max-gb "$MAX_GB" -i "$INTERVAL" --model "$FILTER" >> "$logf" 2>&1 &
        else
            nohup sh "$0" __daemon --dir "$DIR" --days "$DAYS" --max-gb "$MAX_GB" -i "$INTERVAL" >> "$logf" 2>&1 &
        fi
    fi
    echo $! > "$pidf"
    echo "已启动守护进程 pid=$!  日志: $logf"
}

cmd_stop() {
    [ -n "$DIR" ] || die "未指定缓存目录"
    pidf="$DIR/.hicache_admin/daemon.pid"
    if [ ! -f "$pidf" ]; then
        echo "没有找到 pid 文件: $pidf"
        return 0
    fi
    pid=$(cat "$pidf")
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null
        echo "已停止守护进程 pid=$pid"
    else
        echo "进程 $pid 不存在，清理 pid 文件"
    fi
    rm -f "$pidf"
}

cmd_status() {
    [ -n "$DIR" ] || die "未指定缓存目录"
    pidf="$DIR/.hicache_admin/daemon.pid"
    logf="$DIR/.hicache_admin/daemon.log"
    if [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
        echo "守护进程运行中: pid=$(cat "$pidf")"
    else
        echo "守护进程未运行"
    fi
    echo "日志: $logf"
    [ -f "$logf" ] && tail -n 5 "$logf"
}

cmd_log() {
    [ -n "$DIR" ] || die "未指定缓存目录"
    logf="$DIR/.hicache_admin/daemon.log"
    [ -f "$logf" ] || { echo "日志不存在: $logf"; return 0; }
    tail -n "$LINES" "$logf"
}

# ---------------- 参数解析 ----------------
CMD=${1:-help}
[ $# -gt 0 ] && shift
case "$CMD" in
    clean) case "${1:-}" in ''|*[!0-9]*) ;; *) DAYS=$1; shift ;; esac ;;
    size)  case "${1:-}" in ''|*[!0-9]*) ;; *) MAX_GB=$1; shift ;; esac ;;
esac

while [ $# -gt 0 ]; do
    case "$1" in
        --dir|--days|--max-gb|-i|--interval|-n|--lines|--model|-j|--jobs|--date)
            [ $# -ge 2 ] || die "$1 缺少参数值" ;;
    esac
    case "$1" in
        --dir)          DIR=$2; shift 2 ;;
        --days)         DAYS=$2; shift 2 ;;
        --max-gb)       MAX_GB=$2; shift 2 ;;
        -i|--interval)  INTERVAL=$2; shift 2 ;;
        -j|--jobs)      JOBS=$2; shift 2 ;;
        -n|--lines)     LINES=$2; shift 2 ;;
        --model)        FILTER=$2; shift 2 ;;
        --date)         CLEAN_DATE=$2; shift 2 ;;
        --before)       DATE_BEFORE=1; shift ;;
        -f|--force)     FORCE=1; shift ;;
        -h|--help)      usage; exit 0 ;;
        *)              die "未知参数: $1（用 -h 查看帮助）" ;;
    esac
done

case "$INTERVAL" in
    ''|*[!0-9]*) die "--interval 必须是整数秒" ;;
esac
[ "$INTERVAL" -ge 60 ] || INTERVAL=60

case "$DAYS" in
    ''|*[!0-9]*) die "--days 必须是非负整数" ;;
esac
case "$MAX_GB" in
    ''|*[!0-9]*) die "--max-gb 必须是非负整数" ;;
esac
case "$LINES" in
    ''|*[!0-9]*) die "--lines 必须是非负整数" ;;
esac
case "$JOBS" in
    ''|*[!0-9]*) die "--jobs 必须是正整数" ;;
esac
[ "$JOBS" -ge 1 ] || JOBS=1
[ "$JOBS" -gt 64 ] && JOBS=64

case "$CMD" in
    info)       cmd_info ;;
    clean)      cmd_clean ;;
    size)       cmd_size ;;
    orphan)     cmd_orphan ;;
    watch-once) cmd_watch_once ;;
    start)      cmd_start ;;
    stop)       cmd_stop ;;
    status)     cmd_status ;;
    log)        cmd_log ;;
    __daemon)   cmd_daemon ;;
    help|-h|--help) usage ;;
    *)          echo "未知命令: $CMD" >&2; usage; exit 2 ;;
esac
