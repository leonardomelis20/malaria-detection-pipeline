import os
from collections import defaultdict

base = r"C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis"
malariae_path = r"C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis\Malariae"
falciparum_path = r"C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis\Falciparum"
ovale_path = r"C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis\Ovale"
vivax_path = r"C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis\Vivax"
species_paths = [malariae_path, falciparum_path, ovale_path, vivax_path]
species = []
for specie in os.listdir(base):
    img_dir = os.path.join(base, specie, "img")
    if not os.path.isdir(img_dir):
        continue

    prefixes = set()

    for f in os.listdir(img_dir):
        prefix = f.split("-")[0]
        prefixes.add(prefix)
    print(specie, len(prefixes), sorted(prefixes))

    for specie in species_paths:
        img_dir = os.path.join(base, specie, "img")

        if not os.path.isdir(img_dir):
            continue

        counts = defaultdict(int)

        for f in os.listdir(img_dir):
            prefix = f.split("-")[0]
            counts[prefix] += 1
        print(specie, dict(counts))