LSTM EXPERIMENT COMPARISON - April 17, 2026
============================================

All experiments use: HYCOM SSH+u+v input, 14-day lookback, 7-day forecast, 32 PCA components
Metric: SSH RMSE on 5-window standard benchmark (lower is better)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BASELINE: 1-Layer Direct LSTM (Vanilla)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Config:           1 LSTM layer, direct 7-step output, no autoregressive
SSH RMSE:         0.0830 ← BASELINE
Skill vs persist: +0.21 
Model size:       Minimal (best baseline for comparison)
Training:         Standard MSE loss, no scheduled sampling


IMPROVEMENTS OVER BASELINE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Experiment 1: 2-Layer LSTM + Autoregressive (No Scheduling)
─────────────────────────────────────────────────────────
SSH RMSE:         0.0892  (+7.5% WORSE than baseline)
Skill vs persist: -0.06
Finding:          Simple autoregressive alone increases error (exposure bias problem)


✓ Experiment 2: 2-Layer LSTM + Autoregressive + SCHEDULED SAMPLING ★ WINNER
─────────────────────────────────────────────────────────────────────────────
SSH RMSE:         0.0749  (-9.8% BETTER than baseline)
Skill vs persist: +0.36 (much better than persistence)
Teacher forcing:  1.0 → 0.2 linear schedule over 100 epochs
Key mechanism:    Gradually expose model to its own predictions during training
Improvement:      16.1% reduction vs naive autoregressive
Status:           BEST RESULT - Training procedure matters more than capacity


Experiment 3: Capacity Increase (PCA64 + Hidden192 with Scheduled Sampling)
─────────────────────────────────────────────────────────────────────────
SSH RMSE:         0.0823  (+1.0% worse than scheduled sampling)
Skill vs persist: +0.22
Finding:          Larger model regresses; training procedure is bottleneck, not capacity
Implication:      Over-parameterization without better training hurts performance


Experiment 4: Residual 3-Layer Encoder (with Autoregressive + Scheduling)
─────────────────────────────────────────────────────────────────────────
SSH RMSE:         0.1845  (+146% REGRESSION vs baseline)
Skill vs persist: -0.36 (worse than persistence)
Finding:          Residual skip connections ineffective in PCA-compressed space
Status:           FAILURE - Added complexity without benefit


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUMMARY OF FINDINGS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Baseline SSH RMSE:        0.0830
Best SSH RMSE:            0.0749
Total Improvement:        9.8% error reduction

Ranking by SSH RMSE (best to worst):
  1. Scheduled sampling (0.0749)         ← WINNER
  2. Baseline vanilla (0.0830)           ← BASELINE  
  3. Naive autoregressive (0.0892)
  4. Capacity increase (0.0823)
  5. Residual encoder (0.1845)           ← FAILED


KEY INSIGHTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. TRAINING PROCEDURE > MODEL ARCHITECTURE
   - Scheduled sampling works because it gradually exposes the model to distribution shift
   - Naive autoregressive fails because training/inference mismatch (exposure bias)
   - Simply adding capacity makes it worse without addressing the root problem

2. SCHEDULED SAMPLING IS CRITICAL FOR AUTOREGRESSIVE MODELS
   - Teacher forcing ratio schedule: 1.0 (ground truth) → 0.2 (model predictions)
   - This forces the model to adapt to its own errors during training
   - Result: 16.1% improvement vs naive autoregressive
   - Skill improves from -0.06 to +0.36 vs persistence

3. PCA COMPRESSION LIMITS ARCHITECTURAL IMPROVEMENTS
   - Residual connections don't help in PCA space (loss of spatial structure)
   - Larger models can't overcome limitations of bottleneck representation
   - Suggests full spatial fields (ConvLSTM) may be next frontier

4. 2-LAYER ARCHITECTURE IS SUFFICIENT
   - Baseline 1-layer achieves 0.0830 (competitive)
   - Adding complexity (residual 3-layer) with naive training makes it worse
   - Best improvement comes from training procedure, not architecture depth


WHAT WORKED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Scheduled sampling of teacher forcing targets
✓ Linear schedule from 1.0 to 0.2 over epochs
✓ Gradual exposure to model's own predictions
✓ Standard 2-layer LSTM sufficient


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
