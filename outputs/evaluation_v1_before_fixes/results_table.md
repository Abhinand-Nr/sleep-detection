# Evaluation results

## Data split

| clip | split |
|---|---|
| clip01_normal_work | tune |
| clip03a_fake_sleep_face | tune |
| clip04a_head_on_desk | tune |
| clip02_eyes_closed_5s | eval |
| clip03b_fake_sleep_face | eval |
| clip04b_facing_away | eval |
| clip05_phone | eval |
| clip06_dim_light | eval |
| clip08_two_people | eval |

## Tuned parameters (chosen on the tune split only)

- `eyes.closed_ratio` = 0.5
- `motion.still_threshold` = 0.02

## Systems on the **tune** split

| system | seconds | accuracy (all states) | asleep precision | asleep recall | asleep F1 | asleep IoU | event IoU |
|---|---|---|---|---|---|---|---|
| naive | 486 | 0.815 | 0.889 | 0.500 | 0.640 | 0.471 | 0.726 |
| face_only | 486 | 0.679 | 0.908 | 0.494 | 0.640 | 0.470 | 0.715 |
| full | 486 | 0.842 | 0.940 | 0.981 | 0.960 | 0.924 | 1.192 |

## Systems on the **eval** split

| system | seconds | accuracy (all states) | asleep precision | asleep recall | asleep F1 | asleep IoU | event IoU |
|---|---|---|---|---|---|---|---|
| naive | 1161 | 0.675 | 0.570 | 0.545 | 0.557 | 0.386 | 1.502 |
| face_only | 1161 | 0.578 | 0.733 | 0.521 | 0.609 | 0.438 | 1.465 |
| full | 1161 | 0.577 | 0.749 | 0.519 | 0.613 | 0.442 | 1.452 |

## Full system per clip (tuned parameters)

| clip | split | seconds | accuracy | asleep precision | asleep recall | asleep F1 |
|---|---|---|---|---|---|---|
| clip01_normal_work | tune | 122 | 0.910 | 0.000 | 0.000 | 0.000 |
| clip03a_fake_sleep_face | tune | 182 | 0.764 | 0.929 | 0.975 | 0.952 |
| clip04a_head_on_desk | tune | 182 | 0.874 | 0.951 | 0.987 | 0.969 |
| clip02_eyes_closed_5s | eval | 124 | 0.919 | 0.000 | 0.000 | 0.000 |
| clip03b_fake_sleep_face | eval | 182 | 0.857 | 0.970 | 0.990 | 0.980 |
| clip04b_facing_away | eval | 183 | 0.393 | 0.000 | 0.000 | 0.000 |
| clip05_phone | eval | 126 | 0.278 | 0.000 | 0.000 | 0.000 |
| clip06_dim_light | eval | 122 | 0.910 | 0.982 | 0.917 | 0.948 |
| clip08_two_people | eval | 424 | 0.429 | 0.875 | 0.366 | 0.517 |

## Confusion matrix, full system, eval split (rows = labels, columns = predicted)

| label \ predicted | awake | drowsy | asleep | absent |
|---|---|---|---|---|
| awake | 428 | 206 | 74 | 6 |
| drowsy | 0 | 21 | 0 | 0 |
| asleep | 87 | 39 | 221 | 79 |
| absent | 0 | 0 | 0 | 0 |

Metrics are per second. 'asleep' is the positive class; IoU = overlap of predicted and labelled asleep seconds; event IoU = mean IoU per labelled asleep interval.
