# Experiment 1
## Notes
Modified for windows, and for use with CUDA. Generally the architecutre was left unchanged.

## Results
Interpolated frames were just the average of the two neighboring frames, not an interpolation.

## Moving Forward
A new loss function needs to be made which punishes the production of averaged frames.

# Experiment 2
## Notes
Implemented the above punishment through the following code:

```python
loss_warp = torch.tensor(0., device=pred_frame.device)
if warped_0 is not None and warped_1 is not None:
    loss_warp = (self.charbonnier_loss(warped_0, target_frame) +
                    self.charbonnier_loss(warped_1, target_frame))
```

This code compares the warped frames to the target frame, in the hope of nudging the optical flow more toward the target frame instead of the average.

## Results
This did not appear to work. Even when weighing the `loss_warp` quantity by an overwhelming majority, the output was still just the average of the two frames.

After discussing with Claude, it thinks that this is because of two reasons:
1. The L1 loss term is large compared to other losses, so it dominates and forces all weights to 0
2. There is no perceptual loss function, so degenerate solutions which are easy to fall into and hard to get out of cause ruts where the network computes a "good enough" intermediary frame but can't get out to a different slope regime where the results are better.