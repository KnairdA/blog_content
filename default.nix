{ pkgs ? import <nixpkgs> { }, mypkgs ? import <mypkgs> { }, ... }:

pkgs.stdenv.mkDerivation {
  name = "blog.kummerlaender.eu";

  src = pkgs.fetchFromGitHub {
    owner = "KnairdA";
    repo  = "blog.kummerlaender.eu";
    rev    = "a043d5dd1933e4fa9cfa2b10a7fdfa05c6c4d0eb";
    sha256 = "0ykprjw97125miw8pqih3pd8hk2sdc1cginakg0p944svs0p6811";
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
