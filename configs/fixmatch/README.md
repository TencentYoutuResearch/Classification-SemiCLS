# Fixmath reimplementation


## result
| Method | Backbone | batch | gpu num |  dataset |Train iters | sample time per iter | metric|
|:------:|:--------:|:-----:|:--------:|:-------------------:|:--------------:|:-------:|:--------:|
|FixMatch|wideresnet 2x28|64|1|cifar10 l40 |1024|1024| 92.04 +- 2.9  |
|FixMatch pseduo with ema|wideresnet 2x28|64|1|cifar10 l40 |1024|1024| - |
