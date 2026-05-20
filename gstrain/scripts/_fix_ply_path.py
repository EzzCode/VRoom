import pathlib, re

p = pathlib.Path("/home/hussein_essam/gs-workspace/VRoom/gs-train/test_anchor_init.py")
txt = p.read_text(encoding="utf-8")

# Replace the multi-line os.path.join block with a plain absolute string
txt = re.sub(
    r"PLY_PATH\s*=\s*os\.path\.join\(.*?\)",
    'PLY_PATH = "/home/hussein_essam/gs-workspace/my_training_implementation/sparse/0/points3D_text.ply"',
    txt,
    flags=re.DOTALL,
)

p.write_text(txt, encoding="utf-8")
print("PLY path updated.")
