# Task

You have the weights of a language model, `{{model}}`. Make it better at mathematics by editing its
weights directly. Find out what goes wrong inside the network when it fails, identify the components
responsible, and change them in a way you can justify with evidence.

After you finish, the edited and original models are compared on held-out math benchmarks that you never
see. The model answers with these settings: {{generation}}. The last `\boxed{}` answer is graded.

# Your tools

You work in a persistent Python kernel on `{{device}}`. Variables survive between code blocks, and
`torch` and `np` are already imported. To run code, reply with exactly one ```python block and print what
you want to see. Output is cut at {{max_output_chars}} characters, and a block may run for up to
{{cell_timeout_s}} seconds. If the kernel restarts, variables are lost, but saved edits are kept.

These functions are defined:

{{tools}}

You have {{forward_passes}} forward passes in total and up to {{max_turns}} code blocks. {{internet}}

# Rules

- No gradient descent or training of any kind: no backward passes, autograd, optimizers, or iterative
  fitting of weights to a loss.
- Run the model only through `generate` and `read`. Do not re-implement or simulate it.
- Change the model only through `save_edit`. Frozen tensors are refused.
- Do not simply rescale a whole tensor by a constant, and do not make the model skip its reasoning.

# How to work

1. Diagnose first. Read how it fails on practice problems: a wrong answer, no answer, or running out of
   tokens while repeating itself. Go after the largest failure you can explain.
2. Localize causally. Before relying on a component, show that changing it changes the behaviour you
   predicted, at more than one strength.
3. Measure properly. Compare the original and edited weights on the same problems with
   `generate(model="both")`. Accuracy over n samples has a standard error near 50/sqrt(n) points. Keep an edit
   only if its gain clearly beats that on problems you did not use to choose it.
4. Print compact summaries, and only short excerpts of generated text.

When you have made your final choice, reply with no code block. That ends the session. Say what you
changed, what evidence supports it, and why you expect it to help. Your saved edits are what gets evaluated.
