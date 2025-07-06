# LSTM Language Modeling

This project implements and compares several LSTM-based neural network models for language modeling and next-word prediction.

## Models Implemented

1. **LSTM_1L_Normal**

   - Single LSTM layer trained in the normal direction.

2. **LSTM_2L_Normal**

   - Two LSTM layers trained in the normal direction.

3. **LSTM_1L_Reversed**

   - Single LSTM layer trained in reverse sequence order.

4. **LSTM_2L_Reversed**
   - Two LSTM layers trained in reverse sequence order.

## Evaluation

- **Perplexity** was used as the main evaluation metric on Train, Validation, and Test sets.
- Example results:
  - _LSTM_1L_Normal_: Test Perplexity ~234.6
  - _LSTM_2L_Normal_: Test Perplexity ~447.5
  - _LSTM_1L_Reversed_: Test Perplexity ~2424.2
  - _LSTM_2L_Reversed_: Test Perplexity ~877.3

## Example Outputs

- **Sentence generation** at different temperatures (0.1, 1, 10).
- **Next-word prediction** given a seed word.
- **Sentence probability estimation.**
