# NOTE: THIS FLAKE ASSUMES YOU ARE RUNNING NIXOS!
# Although, I do have plans to make it generic in the future.
{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
  outputs =
    {
      self,
      nixpkgs,
      treefmt-nix,
    }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      treefmtEval = treefmt-nix.lib.evalModule pkgs ./treefmt.nix;
    in
    {
      formatter.${system} = treefmtEval.config.build.wrapper;
      checks.${system}.formatting = treefmtEval.config.build.check self;
      devShells.${system}.default = pkgs.mkShell {
        name = "stock-market-prediction";
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
          pkgs.stdenv.cc.cc
          pkgs.zlib
          "/run/opengl-driver"
        ];
        venvDir = ".venv";
        packages = [
          pkgs.nodejs-slim_24
          pkgs.pyright
          pkgs.python313
          pkgs.python313Packages.venvShellHook
          pkgs.uv
        ];
      };
    };
}
