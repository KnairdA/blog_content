{ system ? builtins.currentSystem }:

let
  pkgs    = import <nixpkgs> { inherit system; };
  mypkgs  = import (fetchTarball "https://pkgs.kummerlaender.eu/nixexprs.tar.gz") { };

in pkgs.stdenv.mkDerivation {
  name = "blog.kummerlaender.eu";

  src = pkgs.fetchFromGitHub {
    owner = "KnairdA";
    repo  = "blog.kummerlaender.eu";
    rev    = "b851828bb911d05d82dd36acde6751e7e940b93c";
    sha256 = "0gizi42nvzbrwdxbc5h2rnpa1id1dd3lp7k27pfh8h4px6fcm9qi";
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
