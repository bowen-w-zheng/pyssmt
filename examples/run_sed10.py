from pathlib import Path
from ssmt.cli import main
main([str(Path(__file__).with_name("SED10.mat"))])
