from textwrap import dedent

from llama_cpp import Llama

llm = Llama(model_path="../../../llama.cpp/models/sv13B/stable-vicuna-13B.ggml.q5_1.bin", n_threads=12)

msg = dedent(
    """
    You are pipobot, an arrogant assistant. Answer as concisely as possible.
    <@fainsil:inpt.fr>: Qu'est ce qu'une intégrale de Lebesgue ?
    <@pipobot:inpt.fr>:
    """,
).strip()

print(msg)
print(repr(msg))

output = llm(
    msg,
    max_tokens=100,
    stop=["<@fainsil:inpt.fr>:", "\n"],
    echo=True,
)

print(output)
res = output["choices"][0]["text"]
print(res)
print(res.removeprefix(msg).strip())
