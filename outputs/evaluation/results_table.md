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
| naive | 486 | 0.815 | 0.889 | 0.500 | 0.640 | 0.471 | 0.642 |
| face_only | 486 | 0.679 | 0.908 | 0.494 | 0.640 | 0.470 | 0.593 |
| full | 486 | 0.842 | 0.940 | 0.981 | 0.960 | 0.924 | 0.918 |

## Systems on the **eval** split

| system | seconds | accuracy (all states) | asleep precision | asleep recall | asleep F1 | asleep IoU | event IoU |
|---|---|---|---|---|---|---|---|
| naive | 1161 | 0.796 | 0.581 | 0.962 | 0.724 | 0.568 | 0.912 |
| face_only | 1161 | 0.768 | 0.820 | 0.955 | 0.882 | 0.789 | 0.903 |
| full | 1161 | 0.780 | 0.952 | 0.952 | 0.952 | 0.909 | 0.909 |

## Full system per clip (tuned parameters)

| clip | split | seconds | accuracy | asleep precision | asleep recall | asleep F1 |
|---|---|---|---|---|---|---|
| clip01_normal_work | tune | 122 | 0.910 | 0.000 | 0.000 | 0.000 |
| clip03a_fake_sleep_face | tune | 182 | 0.764 | 0.929 | 0.975 | 0.952 |
| clip04a_head_on_desk | tune | 182 | 0.874 | 0.951 | 0.987 | 0.969 |
| clip02_eyes_closed_5s | eval | 124 | 0.919 | 0.000 | 0.000 | 0.000 |
| clip03b_fake_sleep_face | eval | 182 | 0.857 | 0.970 | 0.990 | 0.980 |
| clip04b_facing_away | eval | 183 | 0.787 | 0.000 | 0.000 | 0.000 |
| clip05_phone | eval | 126 | 0.683 | 0.000 | 0.000 | 0.000 |
| clip06_dim_light | eval | 122 | 0.910 | 0.982 | 0.917 | 0.948 |
| clip08_two_people | eval | 424 | 0.696 | 0.931 | 0.943 | 0.937 |

## Confusion matrix, full system, eval split (rows = labels, columns = predicted)

| label \ predicted | awake | drowsy | asleep |
|---|---|---|---|
| awake | 562 | 225 | 7 |
| drowsy | 0 | 45 | 8 |
| asleep | 10 | 5 | 299 |

Metrics are per second. 'asleep' is the positive class; IoU = overlap of predicted and labelled asleep seconds; event IoU = mean IoU per labelled asleep interval.
