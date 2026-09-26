# Paper reproduction notes

Target paper: *Gradient-Based Learning Applied to Document Recognition* (1998).

The current implementation is a modernized LeNet-style MNIST baseline.
The original architecture and comparison experiments remain unimplemented.

Before running research experiments, split the training data into training and
validation sets. Use validation accuracy to select checkpoints and settings,
and reserve the test set for final evaluation. The current baseline evaluates
the test set every epoch and uses it to select the best checkpoint.

Future work: study the paper and document original versus modern design choices,
then implement and compare them. No experimental results are claimed yet.
