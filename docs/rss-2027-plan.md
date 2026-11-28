# RSS 2027 research plan: Verifying Without Undoing

Proposed subtitle: **Risk-Aware Active Verification of Manipulation Outcomes**.

Status: research proposal, recorded on 27 November 2026. The proposed method,
task families, and experiments below are not implemented or validated results.
The available resources are simulation and a CUDA GPU or cloud GPU budget;
physical robot access is not assumed.

## Research question and motivation

When a robot is unsure whether manipulation succeeded, which observation or
physical check should it perform, accounting for the possibility that checking
destroys a successful outcome or repairs a failed one?

For example, an inserted component can look seated without being mechanically
engaged. A gentle pull might expose the failure or dislodge a correct insertion.
A push might finish an initially incomplete insertion. The verifier must
distinguish evidence about the original outcome from changes caused by checking.

The current Panda workbench supplies useful experimental infrastructure: cameras,
textured objects, physical manipulation, recording, and replay. Its scripted
verification and bounded retry are engineering features, not evidence of this
proposal's novelty or effectiveness. RoboGoGoGo would be the experimental
apparatus; the paper would center on the method and scientific findings.

## Candidate contribution and hypothesis

The proposed contribution is an action-conditioned estimator and verification
policy that jointly model diagnostic observations and changes in task success.
General active perception, verification planning, and risk-aware decision making
are established ideas. Combining them or adding a risk penalty alone would not
establish a research contribution.

**Hypothesis:** at comparable verification time and acceptance coverage,
predicting the effects of checks reduces false acceptance relative to passive
verification while destroying fewer successful outcomes than checks selected
only for information gain.

This is a candidate gap from the reviewed work, not an exhaustive priority claim.
A defensible contribution requires a concrete estimator and selection procedure,
meaningful baselines, and reproducible findings that survive held-out conditions.

## Proposed method

1. Freeze the manipulation controller. A separate verifier receives the task
   instruction, RGB-D, execution history, and realistically noisy proprioception.
   Object poses, contact labels, and simulator success predicates are unavailable
   to it at runtime.
2. Define a bounded set of checks: wait, change viewpoint, withdraw an occluding
   hand, and task-specific gentle physical probes. Include accepting completion
   and abstaining as decisions.
3. Train an action-conditioned model to predict both the observation produced by
   a check and the joint pre-check/post-check success state. Use matched simulator
   branches from the same initial physical state to compare candidate checks.
   Privileged state is permitted for offline labels and evaluation only.
4. Label four outcomes separately: success preserved, success destroyed,
   persistent failure, and failure repaired. Include a matched no-check branch
   over the same time horizon to distinguish check-induced changes from ordinary
   settling or spontaneous failure. Shared randomness and simulator branching
   are experimental tools, not novelty claims themselves.
5. Choose a check by expected reduction in decision error under an estimated
   disruption-risk constraint and a time budget. Update after observing the
   result, then accept, continue, or abstain. A small action-conditioned model
   is the initial scope; training a general video world model or a new VLA is
   not required. Empirical risk estimates are not formal safety guarantees.

The estimator, objective, training losses, and stopping rule must be made precise
before presenting a submission-ready method. This document records the research
direction rather than asserting that those technical details are already solved.

## Closest work and required distinctions

- [PROVE](https://arxiv.org/abs/2611.33733), November 2026 preprint, specifies
  success conditions and evaluates recorded manipulation outcomes. Our proposed
  distinction is choosing checks and modeling their effects on the outcome being
  evaluated. Another camera-based success classifier would be insufficient.
- [How Visible Are Silent Manipulation Failures?](https://arxiv.org/abs/2606.03134),
  June 2026 preprint, studies false-success observability and the inflation
  produced by noiseless simulated proprioception. Passive failure detection and
  a simulator-label disagreement alone are insufficient novelty.
- [ActiveVLA](https://arxiv.org/abs/2601.08325), January 2026 preprint, uses active
  viewpoint selection and 3D zoom for manipulation. Moving a camera alone is
  insufficient; compare against a strong active-view method.
- [RobAuditor](https://ai.uni-bremen.de/papers/kenghagho2025a.pdf), a lab
  manuscript, plans context-adaptive task-verification pipelines. Verification
  planning itself is prior work. The proposed estimator must add value through
  its explicit treatment of diagnosis and physical outcome changes.
- [Backtrack](https://arxiv.org/abs/2610.32941), October 2026 preprint, adds
  lightweight failure detection and recovery to frozen VLAs. Generic retries
  around a frozen controller would be insufficient.
- [RoboCousin](https://arxiv.org/abs/2609.08339), September 2026 preprint, extends
  RoboTwin with user-provided objects and diverse scenes. Object import and asset
  variety are supporting infrastructure rather than this paper's contribution.

These sources delimit the proposal; they do not prove that no other work covers
the intended method. Recheck the closest literature as the method becomes
concrete, including physical probing and constrained active perception.

## Experimental design

Develop three task families, initially with one controlled example each:

- **Holding:** a secure grasp versus an object resting or wedged against the
  hand. A small motion can expose weak retention or dislodge a valid grasp.
- **Placement:** independent stable support versus temporary support from the
  hand or a rim. Withdrawal and bounded probes can expose support dependence
  or destabilize placement.
- **Engagement:** a peg or hooked component visually aligned versus mechanically
  engaged. Pulling can test retention; pushing can repair incomplete engagement.

Specify success by intended downstream function and a declared time horizon
before creating difficult examples. Verify ambiguity under the full observation
history available to the verifier, including realistic sensor noise. Do not
manufacture apparent failures from arbitrary settling thresholds. Record both
the original outcome and the outcome after verification.

Compare fixed waiting, fixed multi-view inspection, a strong passive
vision-language verifier, active viewpoint selection, a fixed physical probe,
and information-gain-only action selection. Give methods comparable sensors,
robot time, and inference budgets. Keep the task controller fixed across
verifiers. If practical, repeat with another controller to check dependence on
one controller's mistakes.

Report false acceptance, acceptance coverage, success destroyed by checking,
failure repaired by checking, verification time and inference cost, and final
task completion. Report risk/coverage/time tradeoffs; abstaining on every case
must not appear to be an improvement. Use paired held-out episodes, multiple
training seeds, and uncertainty intervals. Charge offline data generation and
training costs separately from online verification costs.

Ablate disruption prediction and matched-branch training. Evaluate unseen object
geometry, contact parameters, sensing noise, and latency. Separate training,
selection, and held-out evaluation cases before tuning.

Develop the controlled pilot in RoboGoGoGo/MuJoCo. Plan an independent replication
of the central finding using another physics engine, with aligned task families
and explicit success criteria. That replication is not currently implemented;
cross-engine agreement would not establish real-robot transfer or safety.

## Decision gates and reasons to stop

1. **Pilot by 1 December 2026:** establish physically meaningful ambiguous cases
   and checks that reveal information while sometimes altering outcomes. Revise
   or abandon the hypothesis if waiting or ordinary extra camera views resolve
   all cases, or if ambiguity depends on artificial success thresholds.
2. **Method gate:** demonstrate that learned action selection improves the
   diagnosis/disruption tradeoff over simple fixed rules and information gain
   on held-out states. If not, do not claim a new effective verification method.
3. **Generalization gate:** the central finding must survive realistic sensing
   noise, unseen physical conditions, and independent simulation replication.
   Results depending on privileged runtime state or one engine's contact
   artifact cannot support the intended claim.

Negative findings may still have scientific value if they explain a meaningful
limitation through rigorous experiments. They do not automatically make an
unsuccessful method publishable.

## RSS 2027 submission plan

According to the [official call for papers](https://roboticsconference.org/information/cfp/),
Stage 1 is due **4 December 2026, 23:59 AoE**. The six-page extended abstract has
up to five content pages and one reference page. The problem, hypotheses, and
methodology must be fully developed; experiments may be preliminary. The
**16 April 2027** final-paper deadline applies to invited submissions.
See also the [author FAQ](https://roboticsconference.org/information/author-faq/)
and [reviewer guidelines](https://roboticsconference.org/information/reviewer-guidelines/).

- **28–30 November:** resolve the closest novelty questions and design the
  smallest falsification pilot; specify the estimator and decision procedure.
- **1 December:** assess the pilot gate and decide whether the scientific core
  is defensible enough for Stage 1.
- **2–3 December:** write the methodology, preliminary evidence, limitations,
  and planned experiments. Prepare an anonymized submission only if ready.
- **After Stage 1, if pursuing the direction:** develop the full task families,
  baselines, ablations, and independent replication. Follow any required
  extension charter if invited to Stage 2.

Acceptance is contingent on scientific substance and review. The current demo
does not establish the proposed result. This plan does not authorize claims of
real-robot performance or imply an accepted RSS submission.
