# Setup phase

* [inference_diagnostic_t2v.sh](inference_diagnostic_t2v.sh) runs the inference for t2v, given [24 prompt list](prompts/MovieGenVideoBench_extended_diagnostic_24.txt)
*  [ema_reader.py](ema_reader.py) reads the ema data from a given checkpoint and saves it as a txt file. 
*  [test_attention_inference.py](test_attention_inference.py) for counting number of attention kernel calls. (will not be able to capture
JIT calls)
* [compare_ode_dmd.sh](./compare_ode_dmd.sh) : for the tokyo woman in red, inference comparing ode vs dmd checkpoints
    * The inference of the ode ckpt is ofc in lower quality than the dmd ckpt.
* [validate_both_checkpoints.py](./validate_both.sh) : computes L2 reconstruction loss, by sampling a batch (upto 32)of input noisy latents (sampled from lmdb dataset + noise) and denoising them with the ODE checkpoint and the DMD checkpoint. 
    * For this the config files [ode_validation.yaml](./configs/ode_validation.yaml) and [dmd_validation](./configs/dmd_validation.yaml) were added. There are some unresolvable differences, notably the time step (5.0) and warp denoising used in dmd config, have no equivalent behavior in ODE. So this comparison might not be the correct objective especially for the dmd. 
    * sharp variances in the L2 loss was observed. While the L2 loss for ODE seems reasonable, the ones for DMD seem unreasonable. 
    For batch size 32 
    ```
    --- ODE Init Checkpoint Result ---
    MSE Loss: 0.068848
    Per-sample: [‘0.015991’, ‘0.085449’, ‘0.030273’, ‘0.036865’, ‘0.082031’, ‘0.072266’, ‘0.056152’, ‘0.041016’, ‘0.043457’, ‘0.063965’, ‘0.063477’, ‘0.114258’, ‘0.100586’, ‘0.061279’, ‘0.041992’, ‘0.021606’, ‘0.053711’, ‘0.118164’, ‘0.091309’, ‘0.144531’, ‘0.047363’, ‘0.261719’, ‘0.062988’, ‘0.034180’, ‘0.011841’, ‘0.130859’, ‘0.098145’, ‘0.029541’, ‘0.034424’, ‘0.058838’, ‘0.042725’, ‘0.051025’]

    --- DMD Checkpoint Result ---
    MSE Loss: 0.086426
    Per-sample: [‘0.019287’, ‘0.103027’, ‘0.036377’, ‘0.074707’, ‘0.106934’, ‘0.149414’, ‘0.071289’, ‘0.044434’, ‘0.045654’, ‘0.068848’, ‘0.066895’, ‘0.114746’, ‘0.102051’, ‘0.079102’, ‘0.047363’, ‘0.032227’, ‘0.055420’, ‘0.146484’, ‘0.116699’, ‘0.163086’, ‘0.049072’, ‘0.359375’, ‘0.065430’, ‘0.039062’, ‘0.017822’, ‘0.200195’, ‘0.121094’, ‘0.032471’, ‘0.039307’, ‘0.092285’, ‘0.045898’, ‘0.056396’]
    ```

