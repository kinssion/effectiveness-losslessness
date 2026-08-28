# Model card

## Intended use

Small causal predictive models for controlled research on invertible symbolic
music representations and predictive code length.  The package is intended for
paper verification, checkpoint evaluation, and method replication—not music
generation deployment or quality ranking.

## Training data and ownership

The experiments use user-obtained Pop1K7 and ComMU MIDI under their upstream
terms.  The artifact makes no ownership claim over either corpus or any song.
No raw training data or generated MIDI is distributed.

## Model families

Pop1K7 A–I use 805k–814k active parameters; J uses 689,349 and K 804,675.
ComMU A/D use the corresponding event-model interface.  The intervention is a
coordinate, relation, canonicalization, or carrier operation—not a general
music generator architecture claim.

## Limits

- Predictive code length is not a perceptual-quality guarantee.
- No human evaluation was conducted.
- Public test manifests are already consumed benchmark splits.
- D/H/E/I comparisons do not establish that a pitch prior is universally
  useful; they isolate the stated interfaces on the frozen setup.
- J/K carrier results include intervention-induced vocabulary and compute
  changes.
- The context probe is secondary and lacks its historical window manifest.
- Memorization and copyright risks remain possible for any model trained on
  copyrighted musical corpora; users must perform their own review.

## Checkpoint license

Weights are not part of this Git artifact.  Publication is deferred pending
upstream-license review and will use a separate research/non-commercial notice
if required.  A code license does not determine a weight license.
