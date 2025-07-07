How to create the different input representations:

*ESM2:* https://github.com/facebookresearch/esm \
*SaPROT:* https://github.com/westlake-repl/SaProt \
*PickPocket:* embeddings obtained from in-house model, 4352 dimension \
*Tensor-DTI:* embeddings obtained from in-house model, 256 dimension \

For those obtained representations that are not 256, use `dummy_reduction.py` to reduce the dimension with a dummy linear layer.
