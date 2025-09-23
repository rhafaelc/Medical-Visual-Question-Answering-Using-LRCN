{
  description = "Develop Python on Nix with uv";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs lib.systems.flakeExposed;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.python3
              pkgs.uv
              # System libraries for PyTorch
              pkgs.stdenv.cc.cc.lib
              pkgs.zlib
              pkgs.glibc
            ];

            shellHook = ''
              unset PYTHONPATH
              
              # Set library path for PyTorch
              export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:${pkgs.glibc}/lib:$LD_LIBRARY_PATH"
              
              uv sync
              . .venv/bin/activate
            '';
          };
        }
      );
    };
}
