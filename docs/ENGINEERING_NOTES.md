# Engineering notes

## Why selective prediction?

A closed-set classifier always chooses one of its known labels. In a user-facing plant system that can turn a bicycle,
an unsupported crop, or the wrong plant part into a confident disease. AADS makes abstention part of the typed output
contract and measures false accepts separately from supported-class accuracy.

## Why target adapters?

Crop/part specialists keep updates small and make promotion reversible. The continual objective combines current-task
cross entropy, teacher-feature distillation, bounded replay, and adapter regularization.

## Why two evidence paths?

- The live CUDA path exercises routing, prototypes, adapters, and OOD gates.
- The GPU-free replay validates the frozen acceptance identity and invariants.

The replay is useful for review and CI, but it is never presented as a fresh inference run.
