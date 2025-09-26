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
              pkgs.python312
              pkgs.uv
              pkgs.stdenv.cc.cc.lib
            ];

            env = {
              LD_LIBRARY_PATH = lib.makeLibraryPath [
                pkgs.stdenv.cc.cc.lib
              ];
            };

            shellHook = ''
              unset PYTHONPATH
              export LD_LIBRARY_PATH="${lib.makeLibraryPath [pkgs.stdenv.cc.cc.lib]}:$LD_LIBRARY_PATH"
              uv sync
              . .venv/bin/activate
            '';
          };
        }
      );
    };
}
