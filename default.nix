{ pkgs ? import <nixpkgs> { }, mypkgs ? import <mypkgs> { }, ... }:

pkgs.stdenv.mkDerivation {
  name = "blog.kummerlaender.eu";

  src = pkgs.fetchFromGitHub {
    owner = "KnairdA";
    repo  = "blog.kummerlaender.eu";
    rev    = "7e3246da531228d507734cc6aefa03e9c35c4322";
    sha256 = "043pz9f1lh8albkwxg8q165g5vsg2bw3sjw4cv5bzrbvngcx8r9n";
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
