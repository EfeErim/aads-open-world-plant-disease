# Model card — controlled-demo adapters

## Intended use

Research and portfolio demonstration of selective plant-disease recognition for eight crop/part targets.

## Out of scope

- agricultural treatment decisions;
- autonomous production deployment;
- unsupported crops, parts, diseases, or acquisition conditions;
- using the frozen 48-row acceptance result as a field-performance estimate.

## Safety behavior

The inference contract may return `review` instead of a disease. Unknown crops, unsupported parts, low-margin candidates,
and candidates too close to negative prototypes are blocked before a disease answer.

## Evidence

The frozen CUDA run `20260706T153334Z` passed 48/48 controlled rows: 36 correct disease answers and 12 expected
review/abstain outcomes, with zero negative false accepts and zero wrong-part disease labels.

## Release status

Any public weight release is a checksum-pinned **controlled-demo artifact**. `production_ready=false` is part of the
machine-readable manifest and is enforced by the public downloader.
