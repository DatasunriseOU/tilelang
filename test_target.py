from tvm.target import Target

with Target("cuda"):
    current_target = Target.current()
    print("Current target inside context:", current_target)

print("Current target outside context:", Target.current())
