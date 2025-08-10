import os

folder = "img"

for filename in os.listdir(folder):
    if filename.endswith(".png"):
        name_part = filename.replace(".png", "")
        if name_part.isdigit():
            new_name = f"{int(name_part):04d}.png"
            old_path = os.path.join(folder, filename)
            new_path = os.path.join(folder, new_name)
            os.rename(old_path, new_path)

print("Renaming complete!")
