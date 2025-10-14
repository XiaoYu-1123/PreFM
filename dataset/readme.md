  
```
    dataset
    |-- LLP
    |   |-- data
    |   |   |-- CLAP
    |   |   |-- CLIP
    |   |   `-- st
    |   `-- label
    |       |-- feature_label   # Labels for features
    |       |   |-- audio
    |       |   `-- visual
    |       `-- label           # Event category labels
    |           |-- audio
    |           `-- visual
    |-- UnAV-100
    |   |-- data
    |   |   |-- CLAP
    |   |   |-- CLIP
    |   |   `-- onepeace        # Audio-visual data processed by OnePeace
    |   `-- label
    |       |-- feature_label   # Labels for features
    |       |   |-- test
    |       |   |-- train
    |       |   `-- val
    |       |-- feature_label_modified # Feature labels for PreFM+ on the On AVEL task
    |       |   |-- test
    |       |   |-- train
    |       |   `-- val
    |       |-- label_clip_clap # Event category labels
    |       |   |-- test
    |       |   |-- train
    |       |   `-- val
    |       `-- label_onepeace  # Event category labels for PreFM+ on the On AVEL task
    |           |-- test
    |           |-- train
    |           `-- val
```