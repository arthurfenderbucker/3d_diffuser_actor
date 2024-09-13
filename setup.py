
from setuptools import setup


core_requirements = [
    "einops",
    "torch<2.3",
    "numpy",
    "torchvision",
    "diffusers",
    "dgl",
    "flash_attn<=2.5.3",
]

setup(name='diffuser_actor',
      version='0.1',
      description='3D Diffuser Actor',
      author='Nikolaos Gkanatsios',
      author_email='ngkanats@cs.cmu.edu',
      url='https://nickgkan.github.io/',
      install_requires=core_requirements,
      packages=[
            'diffuser_actor',
      ],
)
   