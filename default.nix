{ system ? builtins.currentSystem }:

let
  pkgs    = import <nixpkgs> { inherit system; };
  mypkgs  = import (fetchTarball "https://pkgs.kummerlaender.eu/nixexprs.tar.gz") { };

in pkgs.stdenv.mkDerivation {
  name = "blog.kummerlaender.eu";

  src = pkgs.fetchFromGitHub {
    owner = "KnairdA";
    repo  = "blog.kummerlaender.eu";
    rev    = "5b981611b8a0d3634898a954bf10cae88a70664c";
    sha256 = "1x7zip5rl411dnvjsv8jqn6i67l1h0k0w2qyraal9xm02rnnlvlw";
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
