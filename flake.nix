{
  description = "blog.kummerlaender.eu";

  inputs = {
    cms.url = git+https://code.kummerlaender.eu/blog.kummerlaender.eu;
  };

  outputs = { self, cms, ... }: {
    defaultPackage.x86_64-linux = cms.generate ./.;
  };
}
