with open("examples/deepseek_mla/example_mla_decode_ws.py", "r") as f:
    text = f.read()

# Replace hardcoded num_stages = 2 with conditional
text = text.replace("num_stages = 2", "num_stages = num_stages if num_stages > 0 else 2")

with open("examples/deepseek_mla/example_mla_decode_ws.py", "w") as f:
    f.write(text)
