# Project Overview

*Fill out this template to help AI understand your project and create an implementation plan.*

## Project Name
<!-- What's your package called? -->
Aura

## Purpose
<!-- In 2-3 sentences, what problem does your package solve? -->
A test best to prototype large-volume Rietveld analysis

## Target Users
<!-- Who will use this? (e.g., biologists, data scientists, yourself) -->
Initially, I will be the only target user. But I have colleagues I would like to share this with. It is not intended to be a production product

## Core Functionality
<!-- What are the 3-5 main things your package should do? -->

1. The package will explore solutions meeting the requirements in 01_requirements_matrix.md
2. It will consider the parameter scope in 02_parameter_scope_matrix.md
3. It will avoid the solutions limited by the killers in 03_prototype_killer_questions.md


## Input/Output
<!-- What kind of data/files does it work with? What does it produce? -->

**Input:** 

* Input will be diffraction-focused datasets. In a format, yet to be determined, this will consist of a set of up to 100 1d histograms for a single measurement that captures sample state. There may be >1000 individual sample states (e.g. where temperature is varied)

* This will be supplemented by an instrument description, typically called an instrument parameter file (of the type all Rietveld programs currently use)

* In addition, a list of crystal phases will be provided (typically via a cif file)

* a system of controlling what parameters in the model are allowed to refine and which are fixed. This is a core challenge with very large numbers of params. We will certainly some automated, scripted way to control params

**Output:** 

* A best fitting Rietveld model for each sample state, that fits all input diffraction data
* visualisations that represent the fit to model relationship for a huge input datasets
* visualisations that allow the user to grok a very large number of parameters and know which ones converged, which diverged, which are not converging 




## Technical Notes
<!-- Any other requirements or constraints? -->


