# Traffic Sign Class Definitions

Adjust these classes to match your actual annotation labels.
The ID numbers must match `configs/dataset.yaml`.

| ID | Name                  | Description                          |
|----|-----------------------|--------------------------------------|
| 0  | stop                  | Octagonal stop sign                  |
| 1  | yield                 | Yield / give way                     |
| 2  | speed_limit_30        | 30 km/h speed limit                  |
| 3  | speed_limit_50        | 50 km/h speed limit                  |
| 4  | speed_limit_60        | 60 km/h speed limit                  |
| 5  | speed_limit_80        | 80 km/h speed limit                  |
| 6  | no_entry              | No entry (red circle + white bar)    |
| 7  | turn_left             | Mandatory left turn                  |
| 8  | turn_right            | Mandatory right turn                 |
| 9  | pedestrian_crossing   | Pedestrian crossing warning          |

## Adding More Classes

1. Add the new class name to `configs/dataset.yaml` under `names:`
2. Increment `nc` in `dataset.yaml`
3. Re-annotate any images containing the new class
4. Re-run `split_dataset.py` and retrain
