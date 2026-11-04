# Results

The headline tables and figures are in the [README](../README.md). This document gives the numbers behind the reading in the [README](../README.md) and the checks made on the strong scores. Scores are defined in the [Metrics document](METRICS.md), the task in the [Problem document](PROBLEM.md) and the methods in the [Methods document](METHODS.md). Unless stated otherwise, the figures are for the full test split, the three-seed means, and the return metrics.

## 1. Return RMSE against naive

- Reduction in return RMSE against naive, seed 0, full test split:

| Method | Reduction |
|---|---:|
| GRU | 6.0% |
| LSTM | 6.0% |
| Transformer | 5.3% |
| LightGBM | 5.0% |
| CNN | 4.9% |
| MLP | 4.8% |
| Ridge | 1.6% |
| DLinear | 1.6% |

- The LSTM minus GRU difference in return RMSE is 0.00002 for seed 0 and 0.00008 averaged over the three seeds. No interval is computed for it, so the two networks are not separated on this evidence.

## 2. Price metrics

- Price metrics move much less than the return metrics. On the 90-stock subset, naive has price MAPE 0.02164, within 0.4% of the best method, GRU (0.02157). LightGBM ties naive at the precision shown.
- Price RMSE is dominated by high-priced tickers.
- The differences between methods are small in absolute terms. The radar figures show relative position within a scope only, and print the raw ranges under each figure.

## 3. Direction accuracy

- Naive has no direction accuracy by construction: it never predicts a sign.
- Of the non-zero test returns, 48.6% are positive, so always predicting a fall scores 51.4% on the full test split (51.5% on the subset). This majority-sign guess is the reference.
- Direction accuracy is measured on the 79% of samples with a non-zero return.

## 4. Scope of the uncertainty

- The seed spread covers training randomness only.
- No interval for the test-set sampling noise is reported, so differences between methods are point values.

## 5. Heavy tails

- Full test split: the 4.5% of samples with an absolute return above 10% carry 55% of the sum of squared returns. This equals the centered variance at this precision, since the mean return is near zero. They also account for 50% to 54% of every learned method's squared error.
- 90-stock subset (seed 0): the 2.9% of samples above 10% carry 43% of the sum of squared returns and 39% to 42% of every learned method's squared error. ARIMA is not in this table.
- Most of the gain over naive comes from these samples. On them, return RMSE falls from 0.147 for naive to 0.131 for GRU. On the other samples it goes from 0.0285 for naive to 0.0283 for GRU.
- The tables in the [README](../README.md) mix both groups and do not report them separately.

## 6. Checks on the strong scores

A return R² of 0.1 and a direction accuracy of 0.59 are high for next-day returns. The cause has not been explained. The following checks were made.

- **Cut-off check.** For eight random cutoff dates between 2015 and 2025 and 26 tickers, every price from the cutoff onward was replaced with random values. The inputs and previous closes of all samples up to the cutoff, and the targets of all earlier samples, were unchanged. This shows that, for those tickers and dates, samples are built without later prices. It does not cover all samples, the model code, or any look-ahead elsewhere in the pipeline.
- **ARIMA check.** The saved ARIMA predictions equal a recomputation with the same function. Its filtered forecasts equal one-step forecasts made from the earlier returns alone (three stocks, 20 days each).
- **Gain by year and gap.** The gain is positive in every year from 2023 to 2026 and in every gap category. Subgroup intervals were not computed.
- **Exchange.** The gain is larger on UPCOM (7% return RMSE reduction for GRU) than on HNX (4%) and HOSE (3%).
- **Size of moves.** For GRU, the reduction is 0.7% on samples with an absolute return up to 10% and 10.6% on larger ones. Among samples that follow a one-day gap, it is 1.4% up to 10% and 12.7% above. Why the networks capture part of the large moves was not examined.
- **Assumed daily limit.** After a one-day gap, 0.16% to 0.71% of simple returns exceed the assumed daily limit by more than 0.5 percentage points, with the highest share on UPCOM. The limits are assumed, not verified. These are exploratory flags, not confirmed violations.
- **Very large returns.** All returns above 50% in absolute value follow a gap of more than four calendar days, and 12% of them reverse on the next day. Their causes remain undetermined; see [Problem document](PROBLEM.md) §6.

## 7. Training budget

- **Main run.** The best boosting rounds of LightGBM were 781, 950 and 872 for seeds 0, 1 and 2, against a limit of 1,000. The number of rounds each run executed is not recorded.
- **Main run, neural networks.** For 11 of the 18 neural runs, the best epoch was the last of the six allowed. This is counted from the logged validation losses, which are printed to four decimals.
- **Longer budget (seed 0 only).** A rerun with 20 epochs and 3,000 boosting rounds stopped each network early on the validation loss. The best epochs, counted from 0, lie between 1 and 8. GRU and Transformer reached epoch 8, which means 12 epochs were run. The best LightGBM round was 781.
- **Effect.** The return RMSE of GRU fell from 0.03918 to 0.03908, a change of 0.00010. Every other method changed by at most 0.00001. The ranking did not change.
- Only one seed was rerun, so the sensitivity of the scores to the budget across seeds was not evaluated.

## 8. Duplicated tickers

- The earlier run, which counted the duplicated tickers twice, is kept with the results (see INDEX.md).
- It differs from the main run by at most 0.00015 in return RMSE on the full test split (CNN, seed 0).
- The subset is a different random draw in that run, so its subset scores are not comparable with the main run.

## 9. Example predictions

- The example figures show HOSE:VCI over its last 60 trading days, from seed 0 of the main run, not the three-seed means above. The predictions file reproduces the seed-0 return RMSE of every method in the results table exactly.
- The stock was chosen from the 90-stock subset without looking at the results.
- One stock over a short range is an illustration, not evidence.

## 10. Demo model

- On the earlier data on which the demo weights were trained, the demo's next-day close error was far larger than naive's. For example, over the last 200 days the MAPE was 0.36 for HNX:BID against 0.015 for the naive forecast.
- The demo's checkpoints have not been re-evaluated on the current data. They should not be used for decisions.
