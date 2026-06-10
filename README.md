# Contribution 1: Tensor Parallelism Support for AffineQuantizedTensor

**Contribution Number:** 1  
**Student:** Praneeth Samineni  
**Issue:** https://github.com/pytorch/ao/issues/988  
**Status:** Phase I In Progress

---

## Why I Chose This Issue

I chose this issue because it connects directly to the type of systems-level machine learning work I want to get better at. The issue involves PyTorch AO, quantization, tensor parallelism, and tensor subclass behavior, which are all important areas for understanding how large models are optimized and scaled in practice.

This contribution also matches my learning goals because it is not just a simple documentation change. It requires reading an unfamiliar open-source codebase, understanding existing test patterns, reproducing failures, and adding support for missing tensor operations until tensor parallelism works for more `AffineQuantizedTensor` quantization types. I hope to learn more about low-precision model execution, distributed model parallelism, and how PyTorch handles custom tensor subclasses internally.

---

## Understanding the Issue

### Problem Description

PyTorch AO currently supports tensor parallelism for some quantization methods, but support is incomplete for all `AffineQuantizedTensor` quantization types. The issue tracks adding tensor parallel support for additional quantization workflows such as float8, int4 weight-only, int8 dynamic activation plus int8 weight-only, uintx weight-only, and fpx-related quantization.

In my own words, the missing piece is that certain quantized tensor types do not yet fully support the operations needed for tensor parallel execution. Tensor parallelism often needs to slice, shard, or redistribute tensors. If the underlying tensor subclass does not implement those operations correctly, the test fails.

### Expected Behavior

Tensor parallelism should work correctly with the supported `AffineQuantizedTensor` quantization methods. When a quantized model or tensor is used in a tensor-parallel workflow, the tensor subclass should support the required operations, and the output should remain correct.

The test file should eventually pass successfully:

```bash
python test/dtypes/test_affine_quantized_tensor_parallel.py
