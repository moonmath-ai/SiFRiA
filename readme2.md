# scripts added
* [inference_diagnostic_t2v.sh](inference_diagnostic_t2v.sh) runs the inference for t2v, given [24 prompt list](prompts/MovieGenVideoBench_extended_diagnostic_24.txt)
*  [ema_reader.py](ema_reader.py) reads the ema data from a given checkpoint and saves it as a txt file. 
*  [test_attention_inference.py](test_attention_inference.py) for counting number of attention kernel calls. (will not be able to capture
JIT calls)
