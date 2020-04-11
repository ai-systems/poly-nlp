from setuptools import find_packages, setup

setup(
    name="NlpTasks",
    version="0.0.1",
    description="WIP",
    url="https://github.com/ai-systems/nlp_tasks",
    author="AI Systems, University of Manchester",
    author_email="mokanarangan.thayaparan@manchester.ac.uk",
    license="MIT",
    packages=find_packages(),
    zip_safe=False,
    package_data={"regra": ["resources/*"]},
    include_package_data=True,
)
