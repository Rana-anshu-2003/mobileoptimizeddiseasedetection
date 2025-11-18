import os

data_dir = "PlantVillage"  # ya jahan tera data tha
classes = sorted(os.listdir(data_dir))

with open("class_names.txt", "w") as f:
    for name in classes:
        f.write(name + "\n")

print("class_names.txt created ✅")
