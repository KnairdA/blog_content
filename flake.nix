{
  description = "blog.kummerlaender.eu";

  inputs = {
    stable.url = github:NixOS/nixpkgs/nixos-21.05;
    cms.url = git+https://code.kummerlaender.eu/blog.kummerlaender.eu;
  };

  outputs = { self, stable, cms, ... }: {
    defaultPackage.x86_64-linux = cms.generate ./.;

    defaultApp.x86_64-linux = let
      system = "x86_64-linux";

      pkgs = import stable { inherit system; };

      serve = pkgs.writeScriptBin "serve" ''
        #!/bin/sh
        pushd ${cms.generate ./.}
        ${pkgs.gatling}/bin/gatling
        popd
      '';
    in {
      type = "app";
      program = "${serve}/bin/serve";
    };
  };
}
