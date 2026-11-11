# Task

You have the weights of a language model, `{{model}}`. Change its behaviour by editing its weights directly:

{{task}}

Find out what inside the network produces the current behaviour, identify the components responsible, and change them
in a way you can justify with evidence. Keep the edit targeted: after you finish, the edited model is also compared with
the original on unrelated math and knowledge questions, and on how much its predictions shift on ordinary text.

# Your tools

You work in a persistent Python kernel on `{{device}}`. Variables survive between code blocks, and `torch` and `np` are
already imported. To run code, reply with exactly one ```python block and print what you want to see. Output is cut at
{{max_output_chars}} characters, and a block may run for up to {{cell_timeout_s}} seconds. If the kernel restarts,
variables are lost, but saved edits are kept.

These functions are defined:

{{tools}}

`practice()` gives practice items, and `practice(split="select")` a separate set for checking an edit before and after.
You have {{forward_passes}} forward passes in total and up to {{max_turns}} code blocks. {{internet}}

# Rules

- No gradient descent or training of any kind: no backward passes, autograd, optimizers, or iterative fitting of
  weights to a loss.
- Run the model only through the provided functions. Do not re-implement or simulate it.
- Change the model only through `save_edit`. Frozen tensors are refused.
- Do not simply rescale a whole tensor by a constant.

# How to work

1. Diagnose first. Look at how the model behaves on practice items before changing anything.
2. Localize causally. Before relying on a component, show that changing it changes the behaviour you predicted, at
   more than one strength.
3. Measure properly. Compare the original and edited weights on the same items, for example with
   `generate(model="both")` on `practice(split="select")`. Accuracy over n samples has a standard error near
   50/sqrt(n) points. Keep an edit only if its gain clearly beats that.
4. Print compact summaries and only short excerpts of generated text. Do not print long lists of raw vocabulary.

When you have made your final choice, reply with no code block. That ends the session. Say what you changed, what
evidence supports it, and why you expect it to help. Your saved edits are what gets evaluated.
