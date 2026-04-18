MODEL EXPERIMENT COMPARISON - April 17, 2026
=============================================

All PCA-LSTM experiments use: HYCOM SSH+u+v input, 14-day lookback, 7-day forecast, 32 PCA components
ConvLSTM operates directly on full spatial fields (no PCA compression)
Metric: SSH RMSE on 5-window standard benchmark (lower is better)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BASELINE: 1-Layer Direct PCA-LSTM (Vanilla)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Config:           1 LSTM layer, direct 7-step output, no autoregressive
SSH RMSE:         0.0830 ← BASELINE
Skill vs persist: +0.21
Model size:       Minimal (best baseline for comparison)
Training:         Standard MSE loss, no scheduled sampling


PCA-LSTM EXPERIMENTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Experiment 1: 2-Layer LSTM + Autoregressive (No Scheduling)
─────────────────────────────────────────────────────────
SSH RMSE:         0.0892  (+7.5% WORSE than baseline)
Skill vs persist: +0.09
Finding:          Simple autoregressive alone increases error (exposure bias problem)


Experiment 2: 2-Layer LSTM + Autoregressive + SCHEDULED SAMPLING
─────────────────────────────────────────────────────────────────
SSH RMSE:         0.0749  (-9.8% BETTER than baseline)
Skill vs persist: +0.36
Teacher forcing:  1.0 → 0.2 linear schedule over 100 epochs
Key mechanism:    Gradually expose model to its own predictions during training
Improvement:      16.1% reduction vs naive autoregressive


Experiment 3: Capacity Increase (PCA64 + Hidden192 with Scheduled Sampling)
─────────────────────────────────────────────────────────────────────────
SSH RMSE:         0.0823  (-1.0% better than baseline, +9.8% worse than sched. sampling)
Skill vs persist: +0.22
Finding:          Larger model regresses; training procedure is bottleneck, not capacity


Experiment 4: Residual 2-Layer Encoder (with Autoregressive + Scheduling)
─────────────────────────────────────────────────────────────────────────
SSH RMSE:         0.0872  (+5.1% WORSE than baseline)
Skill vs persist: +0.13
Finding:          Residual skip connections ineffective in PCA-compressed space


CONVLSTM EXPERIMENT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Experiment 5: ConvLSTM + Autoregressive + Scheduled Sampling ★ CURRENT BEST
─────────────────────────────────────────────────────────────────────────────
SSH RMSE:         0.0635  (-23.5% BETTER than baseline, -15.2% better than best PCA-LSTM)
Skill vs persist: +0.54
Architecture:     Spatial Conv2d gates (no PCA), 2 layers, hidden 64
Teacher forcing:  1.0 → 0.2 linear schedule over 100 epochs
Key mechanism:    Operates on full spatial fields — preserves all spatial structure
Run ID:           20260417T171548Z_ssh-u-v_pca32_h64_e100


Experiment 6: ConvLSTM + Residual Encoder + Autoregressive + Scheduled Sampling
──────────────────────────────────────────────────────────────────────────────
SSH RMSE:         0.0646  (-22.3% better than baseline, +1.7% worse than ConvLSTM baseline)
Skill vs persist: +0.5221
Architecture:     ConvLSTM with zero-init residual skip from last input frame
Teacher forcing:  1.0 → 0.2 linear schedule over 100 epochs
Early stopping:   patience=12, min_delta=0.0001 (best val_loss at epoch 16, train continued to 100)
Key finding:      Residual skip in full spatial space adds no benefit
Run ID:           20260417T211206Z_ssh-u-v_pca32_h64_e100


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUMMARY OF FINDINGS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Baseline SSH RMSE:        0.0830
Best SSH RMSE:            0.0635  (ConvLSTM baseline)
Total Improvement:        23.5% error reduction from baseline

Ranking by SSH RMSE (best to worst):
  1. ConvLSTM + sched. sampling (0.0635)        ← WINNER
  2. ConvLSTM + residual + sched. sampling (0.0646)  (marginally worse)
  3. PCA-LSTM sched. sampling (0.0749)          ← PREVIOUS BEST
  4. Capacity increase PCA64/H192 (0.0823)
  5. Vanilla baseline (0.0830)                  ← BASELINE
  6. Residual encoder (0.0872)
  7. Naive autoregressive (0.0892)


KEY INSIGHTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. FULL SPATIAL FIELDS BEAT PCA COMPRESSION
   - ConvLSTM achieves 0.0635 RMSE vs best PCA-LSTM 0.0749 (15.2% improvement)
   - PCA bottleneck discards spatial structure that is predictive of future SSH
   - ConvLSTM at hidden=64 is smaller parameter count than PCA-LSTM hidden=128

2. SCHEDULED SAMPLING IS CRITICAL FOR AUTOREGRESSIVE MODELS
   - Teacher forcing ratio schedule: 1.0 (ground truth) → 0.2 (model predictions)
   - Works for both PCA-LSTM and ConvLSTM architectures
   - Without it: naive autoregressive is worse than vanilla non-autoregressive

3. TRAINING PROCEDURE > CAPACITY SCALING
   - Larger PCA-LSTM (PCA64/H192) with scheduled sampling: 0.0823 (worse than vanilla)
   - Residual connections add no benefit in PCA space
   - Architecture choice (PCA vs spatial) matters more than capacity

4. CONVLSTM SKILL VS PERSISTENCE: +0.54
   - Substantially better than best PCA-LSTM skill (+0.36)
   - Spatial convolution captures local coherent structures that persist across time


WHAT WORKED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ ConvLSTM operating directly on spatial fields (no PCA)
✓ Scheduled sampling of teacher forcing targets (1.0 → 0.2)
✓ Linear schedule over training epochs
✓ 2-layer architecture sufficient for both model families


WHAT DIDN'T WORK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✗ Naive autoregressive without scheduling (exposure bias)
✗ Larger model capacity without better training (over-parameterization)
✗ Residual encoder topology in PCA space (incompatible representation)
✗ Complex architectures on constrained latent space (diminishing returns)


NEXT DIRECTION: ConvLSTM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Benefits:
  - Operates on full spatial fields (preserves spatial correlations)
  - 2D convolutions respect grid structure naturally
  - No PCA bottleneck; learns spatial features directly
  - Can leverage architectural improvements that failed in PCA space

Hypothesis: ConvLSTM may exceed 0.0749 baseline by 5-15% due to better
spatial representation capacity.
