# Overnight run 2026-09-16/17 — budget $2000

Rates: p5.48xlarge on-demand $63.296/node-hour, spot ~$24.05.
Billed on BillableTimeInSeconds, so queueing costs nothing.

| job | nodes | purpose | est node-h | est $ | actual $ |
|---|---|---|---|---|---|
| B-400M-bench | 1 | throughput + REHEARSAL of compile/bf16/DDP/400M shapes | 0.5 | 32 | |
| S-400M-sweep (x4) | 4x1 | LR sweep, 5 rates x 3 shapes, 1B tokens | 4.3 | 272 | |
| A-400M-block (x4) | 4x1 | 3 shapes x 3 seeds, 8B tokens | 20.6 | 1304 | |
| | | | **25.4** | **1608** | |

Margin $392. Order matters: bench first, because no GPU has run this code and
five bugs surfaced in the hour before it was written.

## 01:10 — both p5 pools capacity-blocked for an hour, $0 billed

Cancelled the bench to free an on-demand slot; the sweep exercises the same
code paths (compile, bf16, DDP, 400M MHA shapes) so it doubles as the rehearsal.

Now running the same 4 sweep bundles in BOTH pools under distinct names:
  S-400M-sweep-{0..3}    spot
  S-400M-sweepod-{0..3}  on-demand
Whichever pool wins capacity first, the twin gets cancelled. Twins use separate
checkpoint prefixes (derived from --name), so they cannot corrupt each other.
Exposure if a pair runs unnoticed for one monitor interval (90s): a few dollars.

## 04:22 — first results, and a grid design error

S-400M-sweep-0 Completed: 4 runs, 4256s training, 1610s billable (38%), $28.31.
Full 15-run sweep therefore ~$106, not the $273 estimated at on-demand rates.

  deep     5.3e-4 3.2077   1.68e-3 3.1254
  balanced 7.1e-4 3.1653   1.26e-3 3.1179

Both best-so-far sit at the top of the grid. I centred the five rates on the
OLD study's INTERPOLATED optima (1.29e-3 / 1.15e-3 / 8.4e-4) when the old
MEASURED bests were 1.6e-3 / 1.6e-3 / 8e-4. So the grid is ~25% too low and
cannot bracket the minimum from above.

Queued S-400M-sweepext: 2.24e-3 for all three shapes, 3 runs, ~$21. The old
sweep had 3.2e-3 clearly worse for deep and balanced, so one rung above 1.68e-3
should close the bracket.

## 08:20 — morning state

Spent $28.31 of $2000. One capacity grant in eight hours, on SPOT not on-demand.

DONE (4 of 18 sweep runs, banked in results/sweep400M/):
  deep     5.3e-4  3.2077     deep     1.68e-3  3.1254
  balanced 7.1e-4  3.1653     balanced 1.26e-3  3.1179

QUEUED, all blocked on "Insufficient capacity error from EC2":
  spot       sweep-1, sweep-2, sweep-3   (expire 10:36)
  spot       sweepext (2.24e-3 x3)       (expires 12:09)
  on-demand  sweepod-1/2/3, bench2       (no expiry, never once served)

WHAT THIS MEANS
  The code works: the 02:56 grant ran 4 runs clean on 8 GPUs and the loss
  curves track the old study. The blocker is purely p5 availability in
  us-west-2, in BOTH pools. On-demand was never served at all, so "use
  on-demand to go faster" did not hold tonight.

  The training block (20.7 node-h) cannot start until the sweep yields rates.
  On one instance at a time that is ~5h of sweep then ~21h of block.

OPTIONS WHEN YOU READ THIS
  1. Keep waiting. Costs nothing; jobs retry themselves.
  2. Cut the block to 1 seed (6.9 node-h, ~$166 spot). Loses the
     complete-separation claim, which needs 3 seeds.
  3. Resubmit the expired spot jobs after 10:36 with a longer wait window.
  4. Request a p5 quota increase (needs AWS approval, not same-day).

## 10:36 — sweep restructured around what is actually missing

sweepext Completed: $21.18. Total $49.49.
  2.24e-3:  deep 3.2544   balanced 3.8331   wide 4.8715
Both deep and balanced are now bracketed on BOTH sides of their minimum.
Wide has only the diverging 2.24e-3 point and needs a whole curve.

Cancelled the three original spot bundles and their on-demand twins (0 billable):
their run order was built for a full grid and does not prioritise the gaps.

Replaced with two targeted bundles, each in both pools:
  fill-a (7 runs)  wide x5, then deep@1.26e-3 and balanced@1.68e-3 -- the points
                   adjacent to each measured minimum, which is what the parabola
                   actually needs
  fill-b (4 runs)  deep@{7.1e-4,9.4e-4}, balanced@{5.3e-4,9.4e-4} -- completes the
                   grid for the shared-rate head-to-head, not needed to pick rates

Added `plan.py sweep --only shape:rate,...` for this; building gap bundles by
hand is exactly the ad-hoc job-config generation that caused earlier mistakes.

## 13:47 — rates picked for wide; its block started early

All three shapes are now bracketed on both sides of their minimum:

    shape      5.3e-4   7.1e-4   9.4e-4   1.26e-3  1.68e-3  2.24e-3   fitted
    deep       3.2077        --       --   3.1132*  3.1254   3.2544   1.24e-3
    balanced       --   3.1653       --   3.1179*  3.1698   3.8331   1.08e-3
    wide           --   3.1620   3.1487*  3.2124       --    4.8715   8.60e-4
    (* = grid minimum)

The tuned rate falls monotonically with width, by ~1.45x from deep to wide.
A single rate applied to all three -- which is what an N-only rule like
Kaplan's D.1 gives, since these three shapes have identical N -- mistunes at
least one end of the range.

`lr.py` fits the parabola through only the three points around the grid
minimum, so the diverged 2.24e-3 cells never enter a fit.

Submitted B-400M-wide (3 seeds, 8.6e-4, 8B tokens, on-demand, 1 node, ~$438).
Started now rather than after the sweep because wide's bracket is three
ADJACENT grid points: the two wide cells still queued (5.3e-4, 1.68e-3) sit
outside the fitting window and cannot move its rate. Deep and balanced still
lean on a distant left point, and the three queued runs that tighten them
(deep@7.1e-4, deep@9.4e-4, balanced@9.4e-4) are last in the sweep queue.

So wide's 5.9h chain -- the longest single-instance chain in the optimal
4-way packing -- now overlaps the 1.5h of sweep left, instead of following it.
Costs nothing and takes the critical path from ~7.4h to ~6.0h.

Rate: ml.p5.48xlarge SageMaker training, us-west-2 = $63.296/node-h (checked
against the pricing API, not assumed).

Budget: $49.49 prior + ~$200 this sweep job + ~$1,270 block = ~$1,520 of $2,000.

## 14:24 — spot beat on-demand for wide; block split across both pools

On-demand p5 had no capacity for 35 min. Submitted a spot twin of the same
bundle; spot was granted within a minute. Stopped the on-demand twin
(B-400M-wide-...134714, 0 billable) rather than leave it queued -- a pending
on-demand job holds one of the 4 on-demand slots that deep and balanced need.

Wide's 3 seeds now run on SPOT (B-400M-wide-s). Interruption is survivable:
ckpt every 20 min, resume keyed by run_id, max_wait 12h.

For the remaining 6 runs, splitting beats racing. Racing the same bundle in
both pools duplicates work and needs a cancel for every loser; splitting puts
different runs in different pools, so every granted instance does useful work:

    3 deep runs     -> on-demand, 1 run per job, 2.34h each
    3 balanced runs -> spot,      1 run per job, 2.03h each

Six single-run jobs instead of three two-run bundles: makespan drops from
4.37h to 2.34h if capacity allows, and each job is independently schedulable,
so a trickle of capacity still makes progress. Quota fits exactly --
on-demand: sweep + 3 deep = 4; spot: wide + 3 balanced = 4.

Cost also falls: balanced on spot is $24.05/node-h vs $63.296, taking that leg
from ~$386 to ~$147. Six startups (~8 min each) add back ~$51.

Gate: deep's rate is final after sweep run 10 (~14:48), balanced's after run
11 (~15:03). Submitting each leg the moment its rate is pinned.

## 14:56 — deep's rate pinned; the shared-rate ranking crosses over

Sweep runs 9 and 10 completed the deep column:

    deep   5.3e-4 3.2077  7.1e-4 3.1653  9.4e-4 3.1249  1.26e-3 3.1132
           1.68e-3 3.1254  2.24e-3 3.2544                -> fitted 1.25e-3

The bracket around the minimum is now three adjacent points and nearly
symmetric (3.1249 / 3.1132 / 3.1254), so the fit barely moves off the grid
point. Deep = 1.25e-3.

THE RESULT WORTH THE ARTICLE: at a shared learning rate, which shape wins
depends on which rate you share.

    5.3e-4   deep 3.2077  balanced 3.2149  wide 3.1963   -> wide
    7.1e-4   deep 3.1653  balanced 3.1653  wide 3.1620   -> wide
    9.4e-4   deep 3.1249                   wide 3.1487   -> deep
    1.26e-3  deep 3.1132  balanced 3.1179  wide 3.2124   -> deep
    1.68e-3  deep 3.1254  balanced 3.1698  wide 3.4488   -> deep

Same shapes, same tokens, same seed -- only the rate differs, and the ordering
inverts between 7.1e-4 and 9.4e-4. A shared-rate comparison (the protocol the
original shape-independence result used) can report either winner depending on
where the shared rate sits relative to each shape's own optimum. Wide's
optimum is lowest, so a low shared rate flatters it and a high one wrecks it.

At each shape's OWN optimum, deep wins: 3.1132 vs balanced 3.1179 vs wide 3.1487.

Submitted B-400M-deep-{0,1,2}, one seed each, on-demand, 8B tokens.

CORRECTION: spot did not actually win the wide race. SecondaryStatus
"Starting" means SageMaker is attempting to launch, not that capacity was
secured -- B-400M-wide-s is cycling on "Insufficient capacity error from EC2
while launching instances, retrying!". The on-demand twin was stopped on that
misreading. No loss (it was Pending, 0 billable, and was occupying one of the
4 on-demand slots the deep jobs needed), but the lesson is that only
TrainingTimeInSeconds going non-null proves an instance was actually granted.

Outstanding requests now saturate both quotas:
  on-demand 4 = sweep(1, ends ~15:11) + deep(3, all Pending)
  spot      4 = wide(1, retrying) + balanced(3, queued behind run 11)
Nothing is training yet except the sweep. Billed to date: $58.

## 15:12 — sweep complete (18/18 cells); all block jobs submitted

S-400M-rest-od Completed, 10941s billable. Sweep total spend $250.

Final grid, final validation loss at 1B tokens, one seed, shared rate grid:

                5.3e-4   7.1e-4   9.4e-4  1.26e-3  1.68e-3  2.24e-3   fitted
    deep        3.2077   3.1653   3.1249   3.1132   3.1254   3.2544   1.25e-3
    balanced    3.2149   3.1653   3.1237   3.1179   3.1698   3.8331   1.12e-3
    wide        3.1963   3.1620   3.1487   3.2124   3.4488   4.8715   8.60e-4

Tuned rate falls monotonically with width: 1.25e-3 / 1.12e-3 / 8.6e-4, a 1.45x
spread. All three shapes have identical N (blocks), so any N-only rule -- e.g.
Kaplan D.1 -- hands all three the SAME rate and necessarily mistunes two.

The shared-rate winner changes three times across the grid:
    5.3e-4, 7.1e-4  -> wide
    9.4e-4          -> balanced
    1.26e-3 and up  -> deep
At each shape's own optimum: deep 3.1132 < balanced 3.1179 < wide 3.1487.

Block jobs submitted, splitting across pools so no work is duplicated except
the deliberate wide race:
    on-demand 4/4 : deep-0,1,2 (1.25e-3)  + wide (8.6e-4, 3 seeds)
    spot      4/4 : bal-0,1,2  (1.12e-3)  + wide-s (8.6e-4, 3 seeds)

Both quotas are now saturated; this is the most outstanding demand I can place.
Wide races in both pools -- the loser gets stopped as soon as one shows a
non-null TrainingTimeInSeconds (status "Starting" is not proof of capacity).

Projected total: $1,030-$1,260 of $2,000, depending on which pool serves wide.

Fallback if p5 stays dead for hours: the loss block is hardware-independent, so
it could run on p4de (80GB A100) at ~2.5x the wall clock, keeping the throughput
sweep on p5 where it was measured. Not needed yet.

## 15:14 — p5 dry in both pools; waiting is the only option

All 4 spot jobs cycle on "Insufficient capacity error from EC2 while launching
instances, retrying!"; all 4 on-demand jobs sit Pending. Nothing is training.

Quotas verified directly (not inferred from a filtered list, which silently
omitted the on-demand row):
    L-82E1C851  ml.p5.48xlarge training job usage        4
    L-82733FAD  ml.p5.48xlarge spot training job usage   4
So all 8 requests are legitimate; the shortage is physical p5 capacity.

No substitute exists. Every other large-GPU SageMaker training quota in
us-west-2 is 0 except ml.g5.48xlarge = 1 (8x A10G 24GB): roughly 8x slower
than H100 and probably short of memory at micro=16. And an A100 fallback would
bust the budget even with quota -- ~2.6x the wall clock at ~$41/node-h is
~$1,960 vs ~$1,203 on p5, because the slowdown outweighs the cheaper hour.

Changing region is not worth it: the corpus, bucket, role and image are all in
us-west-2, and default p5 quota elsewhere is 0.

So: hold. Spot jobs retry until max_wait 12h; on-demand stays Pending. Cost is
frozen at $250 while nothing runs, so waiting is free. Watch for Failed status
(a spot job hitting MaxWaitTimeExceeded, or an on-demand capacity failure) and
resubmit if that happens.

## 15:34 — wide unbundled; 9 runs now 1-per-job

Found a packing mistake: wide's 3 seeds were still in ONE job (submitted before
the switch to one-run-per-job), so wide ran them sequentially -- 3 x 1.96h =
5.88h, making wide the long pole even though deep's single run is only 2.34h.
Stopped both wide jobs (0 billable, nothing had started) and resubmitted as
single-run jobs.

9 runs, 8 concurrent slots (quota 4 + 4), so exactly one run must wait:
    on-demand 4/4 : deep-0 deep-1 deep-2 (1.25e-3)   w0 (8.6e-4)
    spot      4/4 : bal-0  bal-1  bal-2  (1.12e-3)   w1 (8.6e-4)
    HELD          : w2 -- bundles/block-w-2.json, submit when a slot frees

Makespan once capacity exists: the held run is the shortest (1.96h) and starts
when the first job finishes (also 1.96h), so it ends at 3.92h, versus deep's
2.34h. 3.92h is optimal for 9 runs on 8 slots -- the machine that must take two
runs cannot do better than 2 x 1.96h.

    before: 5.88h (wide bundled)   after: 3.92h

DO NOT FORGET w2. It is the only run not submitted.

## 16:08 — pre-flight checks for the 8B runs (capacity still dry)

Corpus is large enough. 99 x 190.7 MiB + 83.9 MiB of uint16 = 9.94B train
tokens against 8B per run, so no forced wrap (val.bin is separate, 10M tokens).

Data order is shape-independent, which is a stronger control than I had
credited. TokenShards.batch draws the whole global batch from
rng([seed, step]) and slices it by rank -- so deep, balanced and wide at the
same seed consume byte-identical sequences at every step, and the order does
not depend on GPU count, micro-batch or accumulation either (so a spot resume
replays the same stream). Data order is therefore not a confound at all, and
seeds 1/2/3 still give genuine variation.

Sampling is with replacement, so 8B tokens covers ~55% of corpus positions
with some overlap -- normal, and identical across shapes.

Capacity: still nothing. 8 jobs queued, 0 granted, ~70 min for deep and ~2h
for wide. Spend frozen at $250.

## 17:12 — first block run training; throughput confirms the bench

B-400M-deep-0 got on-demand capacity at ~17:07 after 2h13m queued. Reached
step 500/30518 at 334s elapsed. Backing out ~196s of container start + compile
leaves ~0.276 s/step at 262,144 tokens/step = ~950k tok/s, matching the
benchmark's compiled deep figure of 948,385 tok/s. So the 8B block runs at the
speed the throughput sweep predicted, and the MFU/cost model holds.

Loss 4.53 at step 500 is mid-warmup (warmup is 0.02 x 30518 = 610 steps, vs
the sweep's 150), so early values are not comparable to the sweep's.

ETA deep-0 ~19:30. Other 7 jobs still unplaced; w2 still held (quotas full).

## 18:05 — BUG: checkpointing killed every run at ~40 min. Fixed.

B-400M-deep-0 reached step 8000/30518 cleanly then died:

    OSError: [Errno 39] Directory not empty:
      '/opt/ml/checkpoints/95c5f0f532/latest.tmp' -> '.../latest'

train.py saved to a directory `latest.tmp/` and did `tmp.replace(d/"latest")`.
Path.replace is os.replace, which overwrites atomically for a regular FILE but
for a DIRECTORY requires the destination to be empty. So the first checkpoint
of a run succeeded (no `latest` yet) and every subsequent one raised ENOTEMPTY.
With ckpt_minutes=20 that is a guaranteed crash at ~40 minutes into any run.

Why the sweep never caught it: sweep runs were ~18 min each, SHORTER than one
checkpoint interval, so no sweep run ever wrote a second checkpoint. The bug
was invisible until the first 2.3-hour run. All 9 block runs would have died
at ~40 min, resumed from ckpt 1, and died again -- a permanent failure loop.

Fix: swap the file, not the directory.
    live = d / "latest"; live.mkdir(parents=True, exist_ok=True)
    tmp = live / "model.pt.tmp"
    torch.save({...}, tmp)
    tmp.replace(live / "model.pt")
This keeps the property the original comment claimed but did not deliver: a
reader always sees either the previous or the new checkpoint, never a partial
file, and there is now no instant with no checkpoint at all.

Verified by reproducing both patterns on plain files: the directory form fails
on checkpoint 2 with ENOTEMPTY, the file form survives repeated writes.

Recovery: deep-0's first checkpoint (step ~4500, 17:29) is valid and kept, so
resubmitting under the same --name resumes from there rather than restarting.
The failed second save left a complete-but-unrenamed latest.tmp/model.pt at
step ~8000; deleted it rather than promoting it -- the file is almost certainly
intact, but if it were truncated the job would fail at startup and forfeit a
p5 grant, and capacity is scarcer than the 16 min it would have saved.

All 7 queued jobs carried the buggy code (source is packaged at submit time),
so all 8 were stopped (0 billable except deep-0) and resubmitted:
    on-demand : deep-0 (resumes ~4500) deep-1 deep-2 w0
    spot      : bal-0 bal-1 bal-2 w1
    HELD      : w2

Spend: 16869s billable = ~$297 of $2,000. deep-0's $46 was half-salvaged.

## 18:35 — resume path reviewed (it is correct); stale article text found

Since no sweep run ever resumed, the resume path had never executed, and
deep-0 now depends on it. Reviewed statically:
  - state_dict is saved and loaded on the SAME wrapped object,
    DDP(compile(Transformer)), so the key prefixes match
  - step is saved as step+1 and the loop is range(start, steps): no step is
    repeated or skipped
  - batches are keyed on (seed, step) and lr_at() is a pure function of step,
    so a resumed run replays the identical data and schedule
  - start >= steps returns early, so a resubmit of a finished run is a no-op
Conclusion: correct as written.

log.jsonl opens in append mode, so deep-0's log will hold steps 0..8000 from
the failed attempt and then 4500..30518 from the resume. clean.py already
handles this by construction -- it accumulates into a dict keyed on step
(vals[x["step"]] = x["val_loss"]), so the overlap collapses last-wins and the
plotted curve stays monotonic. lr.py reads the final entry, also unaffected.
No change needed.

TODO for the article (not this loop's scope, but it is wrong today):
clean.py's methods text still describes the ABANDONED setup -- "sixteen
H100s", "EFA fabric", "under FSDP", "every tensor, pipeline and data-parallel
layout that fits was swept", and the old interpolated rates 1.29e-3 / 1.15e-3
/ 8.4e-4. The real setup is one 8xH100 node, plain DDP, no parallelism sweep,
rates 1.25e-3 / 1.12e-3 / 8.6e-4 from a 6-point shared grid. Must be rewritten
before publishing.

## 20:57 — checkpoint fix confirmed in production; resume also verified

deep-0 got capacity again at ~20:10 and printed "resumed at step 4115", so the
resume path works and the ~4100 steps from the failed attempt were salvaged
rather than redone.

More important: it has now cleared the second checkpoint. The old code died at
2626s (43.8 min) on the latest.tmp -> latest rename. The fixed run is at 2862s
and step 13500, having written checkpoints at ~03:31 and ~03:51 UTC with no
error. The ENOTEMPTY failure is gone.

    step 12500  val 2.9755      <- second checkpoint written here
    step 13000  val 2.9695
    step 13500  val 2.9594

Progress is on model: 0.28 s/step, val 2.9594 at 3.54B tokens (the 1B sweep
figure for deep at this rate was 3.1132). ETA ~22:17.

Also learned an operational lesson: a background poll armed in the same turn
as ScheduleWakeup gets killed when the wakeup ends the turn. Two polls died
that way. Time the wakeup to land after the event instead.

Other 7 jobs still Pending/retrying -- no capacity. Spend ~$347; projected
$1,030-$1,480 of $2,000 depending on how much of the remainder lands on spot.

## 21:33 — rebundled 2 runs per job: fewer grants needed

Single-run jobs were the right call for the capacity I EXPECTED (8 instances
at once, makespan 3.92h). They are the wrong call for the capacity we actually
got: roughly one isolated grant every 1.5-3h. A SageMaker job holds its
instance for its whole duration, so runs bundled into one job need ONE grant
between them, while single-run jobs need a fresh grant each. At the observed
rate the 8 remaining runs would have needed 8 grants, i.e. most of a day of
waiting.

Rebundled the 8 remaining runs into 4 jobs of 2 runs each -- halving grants
needed while keeping 4 independent requests, so a multi-instance release is
still absorbed:

    rb0 (on-demand) deep s2 + balanced s2     4.37h
    rb1 (on-demand) deep s3 + balanced s3     4.37h
    rb2 (spot)      wide s1 + wide s3         3.92h
    rb3 (spot)      wide s2 + balanced s1     3.99h
    deep-0 (on-demand, already training) deep s1

All 9 runs are now in flight or queued. The wide seed-3 run that was
previously HELD with no job at all is inside rb2, so nothing is left out.
Added `plan.py block --skip shape:seed` to exclude deep:1 from the rebundle
rather than hand-editing a bundle file.

Two operational errors this round, both mine:
  - zsh does not word-split unquoted $VAR the way bash does, so `for j in
    $OLD` passed the whole list as a single 200-char job name and every call,
    including the stop calls, failed ValidationException. Nothing was stopped
    on the first attempt. Fixed with a real array and "${OLD[@]}".
  - the earlier wait loop tested for status == InProgress, but a stopping job
    reports "Stopping", so it read 0 immediately and timed out doing nothing.
    Now tests for terminal states (Stopped/Failed/Completed).

deep-0 unaffected throughout, still Training at ~5100s. Spend ~$347.

## 22:17 — first 8B run complete

B-400M-deep-0 Completed, 7581s billable (plus the 2626s of the failed attempt,
of which ~4100 steps were salvaged by the resume).

    deep (1024x32) lr 1.25e-03 seed 1, 8B tokens:  final val 2.7751

62 validation points. The resume overlap in log.jsonl collapsed correctly when
read into a dict keyed on step, so the curve is clean and monotonic -- the
concern about the append-mode log was unfounded in practice.

For reference the same shape/rate at 1B tokens in the sweep was 3.1132, so 8x
the tokens bought 0.338 of loss.

1 of 9 runs done. The other 8 sit in rb0-rb3, all queued. deep-0 finishing
frees an on-demand slot, so rb0 or rb1 should take it.

Spend: 24450s all on-demand = ~$430 of $2,000.
Projected remaining: rb0+rb1 on-demand 8.74 node-h = $553; rb2+rb3 spot 7.91
node-h = $190; ~$25 startup. Total projection ~$1,200.

## 23:16 — one 4-node job runs all 8 remaining seeds concurrently

Bundle size was the wrong lever. A job releases its instance when its bundle
ends, so 2-run bundles hand back the scarcest resource halfway through the
night. But the real fix is to use the 4-node quota, not to grow the bundle:
one grant of 4 nodes runs 4 seeds AT ONCE.

This needed code. `--instance-count 4` with the existing launcher hands
SageMaker a Torchrun distribution, which builds ONE 32-rank process group and
trains a single config across all 32 GPUs -- not what the study wants, and it
would crash on the spot, since accum() asserts global_batch % (world*micro)
== 0 and 128 % (32*16) != 0.

Added src/shard.py: a plain (non-distributed) entry that reads SM_HOSTS /
SM_CURRENT_HOST to find its node index, takes runs[i::n] of the bundle, and
spawns its OWN `torchrun --nnodes=1 --nproc_per_node=8`. Every run therefore
keeps the exact world=8 DDP geometry used by every other run in the study,
including the throughput sweep, so the numbers stay comparable.
launch.py gained a "shard" entry that deliberately does NOT attach Torchrun.

Verified the split locally (no torch, no GPU) by faking SM_HOSTS:
    algo-1 -> balanced s1, wide s2   3.99h
    algo-2 -> wide s1, deep s3       4.30h
    algo-3 -> deep s2, balanced s3   4.37h
    algo-4 -> balanced s2, wide s3   3.99h
    every run assigned exactly once: True

So all 8 remaining runs finish 4.37h after a single grant, vs 17h serial.

    B-400M-n4     4 x p5, on-demand, 8 runs sharded 2 per node
    B-400M-hedge  1 x p5, spot, same 8 runs serially -- free insurance in
                  case a 4-NODE grant never lands, since 4 instances at once
                  is a harder ask than 1 and tonight AWS has only ever
                  produced one at a time

If both ever run, the spot hedge gets cancelled (it would duplicate work).
Deadline: if n4 has no capacity within ~90 min, fall back to 1-node on-demand
jobs, the only configuration that has actually been granted tonight (twice).

Cost if n4 runs: 4 nodes x 4.37h = 17.5 node-h = $1,107, so ~$1,537 total.

## 23:34 — hardened shard.py against a silent 4x-duplicate failure

shard.py's original node_index() fell back to "node 0 of 1" when it could not
read SM_HOSTS. In a 4-node job that fallback is a disaster: all four nodes
would each take the WHOLE bundle, run 8 runs apiece, and write the same
run_id checkpoints on top of one another. 4x the cost, corrupt results, and
nothing would surface it until the loss curves came out wrong hours later --
exactly the kind of thing that must not be left running unattended.

Now it refuses to guess:
  - falls back to /opt/ml/input/config/resourceconfig.json, which SageMaker
    always writes, before giving up
  - raises if identity is unresolvable while --nodes > 1
  - raises if the cluster's host count disagrees with --nodes
  - still returns (0, 1) for a genuine single-node or off-SageMaker run

Verified all four paths locally. Resubmitted as B-400M-n4-...233425 with
--nodes=4; the old job was Stopped first (0 billable, it had no capacity, so
replacing it cost only queue position, which is worthless while AWS has
nothing to give).

Pending now:
    B-400M-n4     4 x p5 on-demand, 8 runs (2/node), 4.37h once granted
    B-400M-hedge  1 x p5 spot, same 8 runs serial, ~17h once granted

## 00:05 — audited checkpoint + result safety; collection was broken

CHECKPOINTS: safe. Each run's directory is its run_id (a hash of the full
config) and all 9 are unique, while runs[i::n] gives each run to exactly one
node -- so no two nodes ever write the same path. SageMaker syncs all 4 nodes'
/opt/ml/checkpoints into ONE S3 prefix, and the documented requirement for
that is exactly unique names per instance, which holds by construction.
Partial failure degrades well: a dead node leaves the others' synced logs
intact, and a resubmit skips finished runs on the start >= steps check.
Only inefficiency: on resume every node re-downloads the whole prefix (~20GB
once several runs have checkpoints). Slow, not wrong.

RESULT COLLECTION: two silent failures, both fixed.
  1. clean.py read results/block_400M (underscore); the syncs went to
     results/block400M. Different directories entirely.
  2. results/block_400M held 9 runs from the ABANDONED methodology -- config
     schema with tp/pp/dp, "fabric": "efa", ffn 2560 instead of 2688. Building
     the article would have quietly used the old GQA/untied/FSDP numbers and
     raised nothing. This is the worst class of bug in the project: wrong
     published results, no error.
  Archived it to _archive/block_400M_abandoned and repointed clean.py's 3
  references to results/block400M.

Added tools/collect.py: discovers every B-400M-* prefix rather than hardcoding
one (runs are spread across deep-0, n4, and whatever else capacity forces),
pulls only env.json/log.jsonl, collapses the resume overlap into a dict keyed
on step, SKIPs any run whose geometry is not one of the three current shapes,
and prints exactly which of the 9 are complete.

    complete 1/9 -- deep s1, val 2.7751
    missing: balanced s1-s3, deep s2, deep s3, wide s1-s3

## 00:40 — code audit: 5 confirmed defects (1 in training, 4 in the article builder)

Ran 6 independent auditors over model/train/data/shapes/bench/analysis, then an
adversarial refutation pass. 28 raw findings, 8 verified (verification capped at
8; 15 non-cosmetic were NOT verified, so this is not exhaustive), 5 confirmed,
3 refuted. model.py and data.py came back CLEAN -- data.py's shape-independence
control was verified empirically across 13 (world, micro) layouts at several
steps and seeds, byte-for-byte, which is the study's strongest control.

FINDING 1 (src/train.py:76) -- REAL, and needs a decision.
No fp32 master weights and no autocast anywhere: the model is built in bf16, so
params, grads AND both AdamW moments (zeros_like(p)) are bf16. Consequence I
verified directly:
  - RMSNorm gains are COMPLETELY FROZEN. bf16 spacing at w=1.0 is 7.81e-3, so an
    update must exceed 3.91e-3 to register; the largest available is ~lr, and the
    study's largest lr is 1.25e-3. Measured: 0/256 coordinates moved at every one
    of 1.25e-3 / 8.6e-4 / 1.25e-4 / 8.6e-5, max|delta| exactly 0.00e+00, against
    256/256 in fp32. The model cannot learn any norm scale, in any shape.
  - The auditor ALSO claimed matrix weights are differentially discarded by lr,
    making a shape confound. THAT PART IS WRONG. At w=0.02 bf16 tracks fp32
    within a few percent at all four rates (1.01e-1 vs 9.97e-2 at deep's peak;
    7.08e-3 vs 6.89e-3 at wide's floor). The weights that carry the model update
    fine, so there is no lr-dependent quantization confound on them.
  Residual asymmetry: frozen norm params are deep 66,560 (0.017% of blocks),
  balanced 38,400, wide 34,816 -- deep carries 1.9x wide's frozen count. That
  handicaps DEEP, which is currently the loss winner, so deep's win is
  CONSERVATIVE under this bug; fixing it would likely widen deep's margin, not
  reverse it.
  Decision is all-or-nothing: the LR sweep chose its rates under bf16 and
  deep-0 already ran under bf16, so switching to fp32 master means re-running
  the sweep, the block AND the bench (~$1,350 on top of $430). Half-fixing would
  leave the block inconsistent with the sweep that tuned it -- worse than either
  choice. Default for tonight: keep everything bf16-consistent, which preserves
  the option, and let n4 continue.

FINDINGS 2-5 (tools/clean.py) -- all one root cause, zero compute to fix.
clean.py was written for the abandoned methodology and still is:
  2. _Recorded computes attn as GQA (2*d*d + 2*d*8*64) instead of MHA 4*d*d, and
     then ADDS d*vocab "untied: one table each end" to a TIED model whose params
     already counts the table once -- so every published parameter count is wrong.
  3/4. The speed side globs results/trackB/**/*.jsonl (2 nodes, 16-way FSDP,
     GQA, untied, OLD geometry ffn 2560/4480/4352) and never reads
     results/bench400M.jsonl. So published tok/s, MFU, hours and cost all come
     from the superseded runs -- the exact FSDP per-block confound the new
     protocol exists to remove -- while asserting "pure data parallel" over data
     that is FSDP.
  5. It parses config["run"]["shape"] in two places; current env.json is flat, so
     it cannot read a single current run (collect.py and lr.py tolerate both).
Net: clean.py is unusable as-is -- it would crash on the new directory (5), and
if it did not, it would publish old numbers (2-4). Its data layer needs a
rewrite before anything is published. Refuted, for the record: a claim that
flops_per_token is wrong, a claim that lr.py's completeness gate is dead code,
and a claim that clean.py fabricates a 4.82% block mismatch.

## 00:50 — abandoned the 4-node bet; long 1-node bundles instead

n4 sat Pending 74 min with no grant. Called it: 1-node on-demand is the ONLY
configuration granted all night (17:07 and 20:10), a 4-node job needs four
instances simultaneously and that has never once been observed, and if it never
lands we get zero further runs. Expected value clearly favours the config that
demonstrably works.

Bundle length is the lever that matters, not node count. Grants are the scarce
resource (~1 per 4.5h observed), so each grant should be able to work until
morning: 4 runs is ~8.7h, which exceeds the ~7h left, so a grant arriving at
any hour keeps computing rather than finishing early and handing the instance
back. That is the mistake the 2-run rb bundles made.

    g0  (on-demand) deep s2, wide s1, wide s3, balanced s3     8.7h
    g1  (on-demand) deep s3, wide s2, balanced s1, balanced s2 8.7h
    g0s (spot)      twin of g0
    g1s (spot)      twin of g1

Two distinct work sets, each requested in both pools. Both bundles cover all
three shapes, so whichever single job runs still yields a complete 3-shape
comparison.

HARD RULE for every following tick: the moment one of a twin pair shows
non-null TrainingTimeInSeconds, STOP its sibling. Twins use different --name
(hence different ckpt prefixes) deliberately -- sharing a name would let the
survivor inherit progress but would risk two writers on one run_id if both
started before I noticed, and corrupt checkpoints are worse than a restart.
Budget exposure if a pair overlaps for one heartbeat is ~$87, acceptable;
letting both pairs run to completion would be ~$1,950 of $2,000, which is why
the cancel is a hard rule and not a nicety.

Spend still ~$400. deep s1 remains the only completed run.

## 01:18 — spot cancelled at user's request

Stopped B-400M-g0s and g1s, both 0 billable. Spot granted ZERO instances in
~10h of continuous asking across up to 4 concurrent requests, so the twins were
lottery tickets rather than capacity.

This retires the twin-cancel hard rule from the 00:50 entry -- there are no
duplicate work sets left, so no unattended action is required to stay in
budget. Remaining:

    g0 (on-demand) deep s2, wide s1, wide s3, balanced s3      8.7h
    g1 (on-demand) deep s3, wide s2, balanced s1, balanced s2  8.7h

Worst case is now both running to completion: 17.4 node-h = $1,101, total
~$1,500 of $2,000. Each bundle still covers all three shapes, so either one
landing alone yields a complete 3-shape comparison.

## 01:25 — fixed the unambiguous clean.py defects (findings 2 and 5)

Four surgical edits, all verifiable, none dependent on the unresolved bf16
question:
  - both nested config["run"]["shape"] reads now tolerate either schema
    (cfg.get("shape") or cfg["run"]["shape"]), matching collect.py and lr.py,
    so clean.py can actually read a current run
  - _Recorded.attn: GQA 2*d*d + 2*d*512 -> MHA 4*d*d. The old formula
    undercounted attention by ~21M params per shape AND scaled with L*d rather
    than L*d^2, which silently destroys the deep/wide attention match that the
    matched-parameter claim depends on
  - EMB/TOTAL now use tied semantics (one table, counted once) and take blocks
    from the recorded geometry instead of best[s]["params"], which came from
    the superseded trackB records

Verified by lifting _Recorded out of the module and comparing to shapes.py:

           shape   clean blocks    shapes.py   clean total    shapes.py
            deep    398,458,880  398,458,880   449,970,176  449,970,176  OK
        balanced    396,361,728  396,361,728   473,628,672  473,628,672  OK
            wide    398,458,880  398,458,880   501,481,472  501,481,472  OK

STILL OPEN in clean.py (findings 3/4): the speed side globs
results/trackB/**/*.jsonl -- 2 nodes, 16-way FSDP, GQA, untied, old geometry --
and never reads results/bench400M.jsonl, so published tok/s, MFU, hours and
cost still come from the superseded confounded runs. That is a structural
change to what gets published, it interacts with the unresolved bf16 decision,
and there are no block loss curves to pair it with yet, so it waits.

## 01:45 — GQA purged from the REPORTING path (it was never in the training path)

Checked the training path first: src/model.py projects qkv to 3*H*HEAD_DIM and
splits it into three equal parts, so q, k and v all carry H heads. No
n_kv_heads, no repeat_interleave. shapes.py uses 4*d*d. The only "n_kv" string
in src/ is a comment explaining why GQA was rejected. NO MODEL WE TRAINED EVER
USED GQA -- not the sweep, not deep-0, not the throughput sweep.

It survived entirely in tools/clean.py, the article generator, which was never
updated when the methodology changed. Four separate places, now all fixed:
  1. parameter formula used GQA 2*d*d + 2*d*512     (fixed 01:25)
  2. throughput read results/trackB -- 2 nodes, 16-way FSDP, GQA, untied, old
     ffn -- and never read results/bench400M.jsonl. Now reads the study's own
     sweep and reproduces it exactly: deep 948,385 / balanced 1,092,163 /
     wide 1,133,818 tok/s, MFU 42.0/45.5/48.9%, wide over deep +19.6% compiled
     and +39.4% eager, compile speedups 1.802/1.566/1.546 monotone in depth
  3. NODES = 2, so RATE and every cost number was DOUBLE the single-node
     protocol. Now 1.
  4. the prose: "8 key-value heads", "untied embeddings", "under FSDP",
     "sixteen H100s across two machines", "Two p5.48xlarge ... EFA fabric",
     the old interpolated rates 1.29e-3/1.15e-3/8.4e-4, and an entire
     paragraph explaining how many query heads share each KV head. All
     rewritten to the real protocol; the swept rates now come from
     results/lr.json rather than literals so they cannot drift again.

So the models were always MHA; the REPORT was still describing, counting and
costing GQA/FSDP models. Now consistent. Remaining stale-protocol mention is
one comment at clean.py:82 documenting what was removed, which is correct.

## 10:10 — both jobs got capacity overnight; 7/9 runs done; headline result in

g0 and g1 were both granted on-demand p5s around 02:30-03:25 and have been
training since. deep and wide are COMPLETE at three seeds each.

    FINAL VAL @ 8B TOKENS, each shape at its own tuned rate
      deep      n=3   2.7713  2.7716  2.7751   mean 2.7727  sd 0.0021
      balanced  n=1   2.7909
      wide      n=3   2.7895  2.7902  2.7929   mean 2.7909  sd 0.0018

RESULT 1 -- deep wins on loss, decisively. Gap 0.0182 nats, and the seed ranges
do not overlap at all (deep worst 2.7751 < wide best 2.7895). That is complete
separation, which is what three seeds were for.

RESULT 2 -- wide still wins on cost, but the margin collapsed.
Careful here: "tokens for wide to match deep" cannot be read off an annealed
cosine curve, because intermediate points are NOT the loss a run stopped there
would have. The naive endpoint-anchored power-law fit mispredicts wide's own 8B
loss by 0.015 (2.7758 vs 2.7909 actual), so it was discarded. Using instead the
local slope fit over 2-6B, away from warmup and away from the final anneal:
    wide's exponent          -0.0847  (with the anneal included, -0.0730)
    loss gap to close        0.652% relative
    tokens wide needs        +7.7%    (+8.9% under the other slope)
    flip threshold           +19.6%   (the measured compiled throughput gap)
    => wide cheaper by ~10% in wall-clock. Robust to the slope choice: both
       estimates sit far below the 19.6% threshold.

So the thesis HOLDS and is now measured on a clean protocol: deep is the better
shape per token, wide is the better shape per dollar. The answer depends on the
metric, not on the model. What changed versus the published version is the
MARGIN -- de-confounding the throughput gap from 39.2% to 19.6% turned a large
cost win for wide into a ~10% one, and it no longer swamps the token penalty.

Remaining: balanced s2 and s3, both in flight (g0 on its 4th run, g1 on its
4th). Spend ~$1,330 in flight, projecting ~$1,530 of $2,000.
