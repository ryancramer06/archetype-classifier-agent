# Dataset summary

Source: Dr. V's CSCI 317H Synthetic Archetype Dataset, version 1.0.0. The supplied
CSV and dataset_card.json are copied unchanged. Permitted use is coursework and
educational demonstration only. The card states personal_data=false.

There are 1,000 generated records. Seed 317 and sklearn.datasets.make_classification
were used with four informative features, one redundant feature, no repeated
features, one cluster per class, class_sep=1.1, shuffle=true, and flip_y=0.04.
Class IDs from abstract feature clusters were mapped to fictional names. A 4%
random reassignment setting does not imply exactly 4% observed labels changed,
because reassignment can select the same class.

## Actual counts and required split

| Fictional class | All | Fraction | Training | Test |
| --- | ---: | ---: | ---: | ---: |
| explorer | 492 | 0.492 | 394 | 98 |
| guardian | 248 | 0.248 | 198 | 50 |
| creator | 150 | 0.150 | 120 | 30 |
| leader | 110 | 0.110 | 88 | 22 |
| Total | 1000 | 1.000 | 800 | 200 |

The split is train_test_split(test_size=0.20, stratify=y, random_state=317).
There are no missing values or duplicate feature rows. The loader checks the
supplied schema, ranges, labels, row count, and exact class counts before fitting.
Hashes in models/metadata.json identify the original CSV/card. The dataset is
never regenerated or relabeled by this project.

## Six predictors, in this order

| Position | Name | Unit | Inclusive range |
| ---: | --- | --- | --- |
| 1 | palm_width | synthetic_cm | 6.0-11.0 |
| 2 | palm_length | synthetic_cm | 14.0-23.0 |
| 3 | finger_ratio | dimensionless | 0.75-1.15 |
| 4 | face_width | synthetic_cm | 11.0-18.0 |
| 5 | face_height | synthetic_cm | 15.0-25.0 |
| 6 | eye_distance | synthetic_cm | 4.5-7.5 |

finger_ratio is a fictional index-length/ring-length ratio; component lengths
are not separately modeled. The target is archetype. Target values and internal
row indices are never predictors. The programmatic feature vector follows this
order regardless of the key order in a JSON request.

The generator maps an abstract value x using u=0.5*(1+tanh(x/3)), then maps u to
each feature's teaching range. This range mapping is part of data generation,
not training-fold standardization. The fitted StandardScaler still learns its
means and population standard deviations from training data only.

The names explorer, guardian, creator, and leader describe fictional narrative
themes. They are not measured personality, identity, health, demographic,
relationship, ability, or future-behavior characteristics. No biological link is
validated. These numeric fields should not be populated with personal measurements
or used to assess real people.

Inference rejects missing, nonnumeric, Boolean, nonfinite, out-of-range, and
extra features. No imputer fills missing inputs. There are no personal images,
measurements, or external face/palm datasets in this project.
