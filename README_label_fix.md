# Label Value Error Fix

## Problem

The error message indicates that the model is encountering a label value of 36, which is outside the valid range of [0, 36). This means that during training or prediction, the model is trying to use a class index that doesn't exist in the defined classes.

```
Received a label value of 36 which is outside the valid range of [0, 36). Label values: 9 4 18 8 8 0 24 8 5 8 3 20 20 12 19 22 36 13 36 6 11 15 17 31 24 0 30 2 8 25 4 36
[[{{node compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits}}]] [Op:__inference_multi_step_on_iterator_2755]
```

## Root Cause

After investigation, the issue was identified as the inclusion of a `templates` directory in the training data. This directory is not a valid class but was being treated as one during training, resulting in an index of 36 (outside the valid range of 0-35 for the 36 actual classes).

## Fix

The following changes were made to fix the issue:

1. Modified `fix_model.py` to exclude the `templates` directory when preparing training data:
   ```python
   # Skip non-directories and the templates directory
   if os.path.isdir(class_dir) and class_name != "templates":
   ```

2. Modified `train_expanded_model.py` to explicitly specify the classes to use, excluding the `templates` directory:
   ```python
   classes=[d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)) and d != "templates"]
   ```

3. Removed the existing `templates` directory from the `temp_train_data` directory.

## How to Test

To verify the fix:

1. Run the fixed model training script:
   ```
   python fix_model.py
   ```

2. Or run the expanded model training script:
   ```
   python train_expanded_model.py
   ```

The training should now complete without the "label value outside valid range" error.

## Prevention

To prevent similar issues in the future:

1. Always validate the directory structure before training to ensure only valid class directories are included.
2. Use explicit class lists rather than relying on directory scanning when possible.
3. Add validation checks to identify and warn about potential non-class directories before training begins.