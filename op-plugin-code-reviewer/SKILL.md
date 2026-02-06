---
name: code-review
description: Code review checklist for MindSpore op_plugin merges. Use when reviewing `mindspore_op_plugin` operator implementations and tests, including coding rules, functional test coverage (vmap/bf16/non-contiguous/dynamic shape), and performance criteria.
---

# MindSpore Op Plugin Code Review

## Workflow
1. Locate the change scope (operator implementation, functional tests, performance tests).
2. Check coding rules.
3. Check functional test requirements and coverage items.
4. Check performance requirements.
5. Report issues and recommendations.

## Coding Rules
- Ensure there are no Chinese comments in code.
- Ensure copyright year is 2026.
- Do not use try/except to skip unsupported cases.

## Kernel .cc Review Rules
- When reviewing `kernel/*.cc`, confirm the called ATen operator under ` mindspore_op_plugin/third_party/libtorch/include/ATen `; if the same-named ATen operator has an `_out` variant, prefer the `_out` variant.
- Do not add unnecessary validations.
- Do not add unnecessary forced type casts (for example, `.to()`).

## Functional Test Baseline Requirements
- Use `level_mark='level1'` for vmap cases, `level_mark='level0'` for others.
- Decorator template: `@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')`
- Parametrize mode: `@pytest.mark.parametrize('mode', ['pynative', 'KBK'])`
- Each test case must cover both forward and backward.
- Each test case must cover both `pynative` and `KBK`.
- For `KBK` mode, wrap the call with `jit(..., backend="ms_backend", jit_level="O0")`, for example:

```python
if mode == 'pynative':
    output = arange_forward_func(start, end, step)
else:
    output = jit(arange_forward_func, backend="ms_backend",
                 jit_level="O0")(start, end, step)
```
- `rtol` and `atol` must both be 0.
- Do not test scenarios unsupported by PyTorch.

## Non-Contiguous Input Construction
- Use `mint.transpose` to construct non-contiguous inputs.
- Example:

```python
def test_sign_non_contiguous(mode):
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    dtype = np.float32
    x = generate_random_input((2, 3, 4), dtype)
    pt_x_base = torch.tensor(x)
    pt_x = pt_x_base.transpose(0, 1)
    assert not pt_x.is_contiguous()

    ms_x_base = ms.Tensor(x)
    ms_x = mint.transpose(ms_x_base, 0, 1)
    assert not ms_x.is_contiguous()

    expect = generate_expect_forward_output(pt_x)
    output = sign_forward(mode, ms_x)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
    grad = np.ones(expect.shape, dtype=expect.numpy().dtype)
    expect_grad = generate_expect_backward_output(pt_x, torch.tensor(grad))
    output_grad = sign_backward(mode, ms_x)
    allclose_nparray(expect_grad.detach().numpy(), output_grad.asnumpy(), equal_nan=True)
```

## Functional Test Coverage Checklist
- Validate default-parameter scenario.
- Validate empty tensor.
- Validate `inf` and `nan`.
- Supported dtypes aligned with `pytorch_cpu`.
- Validate input value ranges.
- Cover input dimensions 0D-8D.
- Cover all supported dtypes.
- Support implicit type promotion.
- Support broadcasting.
- Validate input constraints.
- Forward accuracy verification passes.
- Backward verification.
- Support dynamic shape/rank/attributes.
- All interface-related test cases in test repo pass.
- Support `bf16`.
- Support non-contiguous inputs.
- Forward/backward results match `torch_cpu` with zero diff.
- For multiple inputs, allow different dtypes per tensor.
- If vmap is supported, provide batch 8,16,32,64,128 cases (level 1).

## Performance Test Requirements
- Forward end-to-end time minus framework noise <= 1.1x `pytorch_cpu`.
- Backward kernel total time < 1.1x `pytorch_cpu`.
