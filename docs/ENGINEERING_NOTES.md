# Engineering notes

## Why selective prediction?

A closed-set classifier always chooses one of its known labels. In a plant-disease system that can turn a bicycle, an
unsupported crop, or the wrong plant part into a confident disease. AADS makes abstention part of the output contract
and measures negative false accepts separately from supported-class accuracy.

The public policy rejects missing candidates, unsupported targets, non-finite scores, low similarity, low margin,
negative-prototype conflicts, and empty disease labels before returning `accept`.

## Why target adapters?

Crop/part specialists keep updates small and make rollback target-specific. The continual objective combines
current-task cross entropy, teacher-feature distillation, bounded reservoir replay, and adapter regularization.
Public code validates label ranges, tensor shapes, finite numeric inputs, and non-negative loss weights so malformed
training state cannot silently produce a plausible loss.

## What “replay” means here

There are two evidence paths:

- The archived CUDA path ran routing, prototypes, adapters, and OOD gates.
- The public GPU-free path validates immutable identifiers and recomputes the accepted summary from 48 sanitized
  recorded decisions.

The second path catches missing, altered, duplicated, or internally inconsistent evidence. It is not fresh inference
and cannot establish generalization.

## Why publish failed readiness artifacts?

Artifact integrity and model readiness are different gates. The release proves that a reviewer can anonymously fetch
the exact bounded files named by an immutable manifest and verify every SHA-256 digest. The accompanying readiness
records prove that none of the eight adapters meets the autonomous deployment policy. Keeping both facts visible is
more useful than hiding failed experiments behind a successful controlled demo.

## Public/private boundary

The full research repository and datasets remain private. This public edition excludes source images, local paths,
credentials, unreviewed dataset licenses, large run histories, and internal orchestration. The row exporter preserves
only decision fields needed to audit the frozen controlled-demo summary.
