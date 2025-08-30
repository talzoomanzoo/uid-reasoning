import pkgutil, importlib, sys
pkgs = []
for name in ["torch","torchvision","torchaudio"]:
    m = importlib.import_module(name)
    import pkg_resources as pr
    dist = pr.get_distribution(name)
    print(dist.project_name, dist.version)

