# summarize_bench.awk
#
# Fix vs. the original: acc/rmsle/rounds/tokens/runtime/retries/failed were only
# ever being SET (last-write-wins) inside each "BENCHMARK COMPLETED" block, but
# each module run has 3 such blocks (one per law_version) before its one
# "Total time" line. The original script's printed row therefore reflected only
# whichever law_version happened to run last -- e.g. a module with law
# versions scoring 100%/100%/0% could print "Acc: 0.00%" and look like a total
# failure when it wasn't.
#
# This version accumulates every "BENCHMARK COMPLETED" block seen since the
# last "Total time" line, then at "Total time" it AVERAGES the per-trial
# metrics (acc, rmsle, rounds, tokens) across however many law-version blocks
# were seen -- the natural aggregate, since each block is itself already an
# average over that version's 4 trials, so this is equivalent to averaging
# over all 12 trials -- and SUMS the block-level counters (runtime, retries,
# failed), since those are genuinely additive across law versions rather than
# things you'd want to average.
#
# If a module's "Total time" line fires with ZERO new BENCHMARK COMPLETED
# blocks since the last reset (i.e. every one of its configs was already
# complete from an earlier session and nothing new ran), the row prints
# "ALREADY COMPLETE (no new trials this session)" instead of misleading
# 0.00%/0.0000 values that would otherwise look like a real bad result.

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
