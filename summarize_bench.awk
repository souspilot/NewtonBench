/Agent Backend:/ {
    backend = $NF
}
/Module:/ {
    module = $NF
}
/BENCHMARK COMPLETED/ {
    in_benchmark = 1
    next
}
in_benchmark && /Average Exact Accuracy:/ {
    v = $NF
    gsub(/%/, "", v)
    acc_sum += v; acc_n++
}
in_benchmark && /Average Raw RMSLE:/ {
    rmsle_sum += $NF; rmsle_n++
}
in_benchmark && /Average Total Tokens Used per Trial:/ {
    tokens_sum += $NF; tokens_n++
}
in_benchmark && /Average Rounds to Completion:/ {
    rounds_sum += $NF; rounds_n++
}
in_benchmark && /Total Runtime:/ {
    runtime_sum += $(NF - 1)  # "Total Runtime: 1149.52 seconds" -> second-to-last field
}
in_benchmark && /Failed Trials \(after all retries\):/ {
    failed_sum += $NF
}
in_benchmark && /Total Retry Attempts:/ {
    retries_sum += $NF
}
/⏱ Total time:/ {
    wall = $(NF - 1) "m"

    if (acc_n == 0) {
        printf "%-22s %-20s %7s  ALREADY COMPLETE (no new trials this session)\n",
               backend, module, wall
    } else {
        acc = acc_sum / acc_n
        rmsle = (rmsle_n > 0) ? rmsle_sum / rmsle_n : 0
        rounds = (rounds_n > 0) ? rounds_sum / rounds_n : 0
        tokens = (tokens_n > 0) ? tokens_sum / tokens_n : 0
        printf "%-22s %-20s %7s  Acc:%7.2f%%  RMSLE:%8.4f  Rounds:%6.2f  Tokens:%9.2f  Retries:%3d  Failed:%2d  Runtime:%9.2fs  (avg over %d law version%s)\n",
               backend, module, wall, acc, rmsle, rounds, tokens, retries_sum, failed_sum, runtime_sum,
               acc_n, (acc_n == 1 ? "" : "s")
    }

    completed[module "|" backend] = 1
    in_benchmark = 0
    acc_sum = rmsle_sum = tokens_sum = rounds_sum = runtime_sum = retries_sum = failed_sum = 0
    acc_n = rmsle_n = tokens_n = rounds_n = 0
}
END {
    key = module "|" backend
    if (!(key in completed) && module != "" && backend != "") {
        printf "%-22s %-20s RUNNING (%d law version%s completed so far this session)\n",
               backend, module, acc_n, (acc_n == 1 ? "" : "s")
    }
}
