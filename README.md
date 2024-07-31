# AFMfit

A python package for flexible fitting of AFM images [1].

## Installation

AFMfit can be installed with _pip_ package manager :

```
git clone https://gricad-gitlab.univ-grenoble-alpes.fr/GruLab/afmfit.git
pip3 install afmfit
```
_(Computation time : 1 min on a 16-core personal desktop)_

**Recommendation :**

AFMfit uses [ChimeraX](https://www.cgl.ucsf.edu/chimerax) for molecular visualizations. 
It is recommended to have ```chimerax``` installed and available in the PATH variable. 

## Examples

Examples are available in the form of Jupyter notebooks.
Make sure you have [Jupyter](https://jupyter.org/) installed : 
```
pip3 install jupyter
```
The tutorial implements the data analysis of synthetic AFM images of Elongation Factor 2 (EF2) described in [1] and can be launched by :
```
python3 -m notebook examples/afmfit_tuto.ipynb
```
_(Computation time : 5 min on a 16-core personal desktop)_

## <span style="color:cyan"> *NEW*</span> - GUI

You can now use most of AFMFit features through the GUI. Make you have [Tkinter](https://docs.python.org/3/library/tkinter.html#module-tkinter) installed. For example (Ubuntu) :
```
sudo apt install python3-tk
```
Then you can simply run the script:
```
/path/to/afmfit/afmfit_gui
```

## Authors

Rémi Vuillemot

LJK - Université Grenoble Alpes 

e-mail: remi.vuillemot@univ-grenoble-alpes.fr

## Citations

[1] Vuillemot, R., Pellequer, J. L., & Grudinin, S. (2024). AFMfit: Deciphering conformational dynamics in AFM data using fast nonlinear NMA and FFT-based search. bioRxiv, 2024-06.
