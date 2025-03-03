# latent-reasoning

## Design principal
1. topp and topk are fixed in model. Ensure the searching space is not to huge.
2. temperature should be considered in generation and sampling, but could be relaxed while training.
3. the model should be fed input ids or a tuple of ids with its weight.