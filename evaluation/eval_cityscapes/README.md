# Standard metrics for Cityscapes

## Steps

- Train and Test
- use `rgb2label` scripts to transform the rgb to ind label
- run `bash ./evaluate.sh`

## Tips

- Same examples are given in file `real_B` and `fake_B`
- You can run the scripts `evaluate.sh` directly to make sure the mean pixel accuracy is 100% in advance.
- The labels in `real_B` and `fake_B` should be resized to 256 x 128 as given example.
- The performance of our pre-trained model is shown in `evaluation_results.txt`