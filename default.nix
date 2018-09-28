{ system ? builtins.currentSystem }:

let
  pkgs    = import <nixpkgs> { inherit system; };
  mypkgs  = import (fetchTarball "https://pkgs.kummerlaender.eu/nixexprs.tar.gz") { };

in pkgs.stdenv.mkDerivation {
  name = "blog.kummerlaender.eu";

  src = pkgs.fetchFromGitHub {
    owner = "KnairdA";
    repo  = "blog.kummerlaender.eu";
    rev    = "c072eaa95c783b9c5d26535cc22694c7a62eb6cb";
    sha256 = "0iqjwixzd6aks9ir2ij33d8qmd3s0h28frh7r7xw7ylbrli86xag";
  };

  LANG = "en_US.UTF-8";

  buildInputs = [
    pkgs.pandoc
    pkgs.highlight
    mypkgs.katex-wrapper
    mypkgs.make-xslt
  ];

  installPhase = ''
    mkdir source/00_content
    cp -r ${./articles} source/00_content/articles
    cp -r ${./tags}     source/00_content/tags
    cp    ${./meta.xml} source/00_content/meta.xml

    make-xslt
    mkdir $out
    cp -Lr target/99_result/* $out
  '';
}
