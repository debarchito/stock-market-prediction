{
  projectRootFile = "flake.nix";
  programs = {
    isort.enable = true;
    nixfmt.enable = true;
    ruff-format.enable = true;
    ruff-check.enable = true;
    taplo.enable = true;
  };
}
