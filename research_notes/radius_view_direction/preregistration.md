# Pre-registration: is the within-image radius mostly view direction?

Branch: `radius-view-direction` (stacked on `radius-aug-regression`, PR #49).
Written before any analysis code, 2026-10-30.

## Question

In `radius_aug_regression` the 13 augmentation parameters explained only 6-13% of
the within-image variance of `log ||z||`, and the leftover radius still predicted
view correctness (within-image AUC 0.55-0.56). Is that leftover radius the view's
direction (the radius is ~95% predictable from direction), crop content, or
something else?

## Data

The nine `augreg_radius_marginal_lam{0.02,0.05,0.1}_s{0,1,2}` dumps already on disk
(8 global views per STL-10 test image). No re-embedding. Direction `u = z/||z||`.

Measured before registering (descriptive, not a prediction): the angle between two
views of one image is 18 / 22 / 24 degrees at lambda 0.02 / 0.05 / 0.1, against
about 89 degrees between different images.

## Model

Same within-image fixed-effects model as `radius_aug_regression`,
`y_iv = alpha_i + beta' x_iv + eps_iv`, `y = log ||z||`, with added blocks:

- **direction**: (a) the angle of `u_iv` to the image's leave-one-out mean
  direction (the other 7 views); (b) `f(u_iv)`, an image-grouped 5-fold
  cross-fitted ridge prediction of `y` from `u`.
- **content**: for the crop box (saved `top/left/height/width/flip`) on the raw
  STL-10 test pixels: mean luminance, luminance sd, mean gradient magnitude, and
  distance of the crop's mean color from the whole image's mean color.

Reported: cross-validated within-image R2 (image-grouped) for params, params +
direction, params + content, params + both, and direction alone. Direction is
reported both added after params and entered first, since it is not independent of
them.

Part C: within-image AUC for probe correctness of the radius residual after each
model, with the same bootstrap over images.

## Predictions

- **V1**: params + direction raises CV within-R2 by at least 0.20 over params
  alone in at least 7 of 9 checkpoints. (Direction is where the radius lives.)
- **V2**: content adds at most 0.03 over params + direction.
- **V3**: the residual after params + direction + content keeps less than half of
  the AUC lift over 0.5 that the residual after params alone had (0.55-0.56),
  i.e. AUC below 0.53, in at least 7 of 9 checkpoints.
- **V4**: the direction-block effect is larger at higher lambda (the view spread
  grows with lambda); sign-checked across seeds.

## Reading rules

- V1 and V3 hold: the within-image radius is mostly view direction and its
  correctness signal is a direction effect, not a separate quantity.
- V1 holds, V3 fails: direction explains the radius variance but the radius keeps
  correctness information beyond direction and content.
- V1 fails: the within-image radius is largely independent of the direction;
  look at other sources (local crops, network state).

Limits to state in the note: global views only, one dataset, one backbone, three
seeds per lambda.
