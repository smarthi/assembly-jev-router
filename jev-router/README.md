# Jev-style classifier/router in Apple Silicon assembly

This is a small, local decision primitive inspired by Jev's typed-decision
interface. It is **not** an implementation of TypeSafe AI's proprietary model.
It provides the part that can be implemented without their trained model:

- predefined, type-safe classes;
- signed weighted classification over an integer feature vector;
- best and runner-up scores;
- a confidence value in basis points;
- routing through a class-indexed function table.

The classifier is deterministic and allocation-free. The caller supplies the
model, feature vector, output structure, and optional route handlers.

## Build and run

```sh
clang -Wall -Wextra -Werror jev_router.s test_jev_router.c -o test_jev_router
./test_jev_router
```

## Model layout

Weights are signed 32-bit integers in class-major order. For `C` classes and
`F` features, provide `C * F` weights. Classification computes:

```text
score[class] = bias[class] + sum(weights[class, feature] * features[feature])
```

The highest-scoring class wins. Ties select the lowest class ID. A production
system can put a text encoder, embedding model, sensor decoder, or ordinary
feature-extraction code in front of this primitive.

Confidence is a transparent score-margin heuristic, not a statistically
calibrated probability. Real calibration requires validation data and a fitted
calibration method; Jev's learned language understanding and calibration cannot
be reproduced from its public API description alone.
