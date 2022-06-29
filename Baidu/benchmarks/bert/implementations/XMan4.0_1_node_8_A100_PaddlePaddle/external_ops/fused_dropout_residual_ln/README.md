# function
This fused op implements the following logics:
```
layer_norm(residual + dropout(input))
```

# Accuracy test
python test_fused_dropout_op.py

# Perf test
python test_fused_dropout_perf.py


# TODO
1. In the unittest, we only test the accuracy when dropout_rate is 0.
2. how to set is_test is true for dropout in eval phase?
