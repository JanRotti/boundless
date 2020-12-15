    Synthetic GPR Image Generation using Generative Adversarial Networks
    Copyright (C) 2020  Jan Rottmayer

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
    
    
    
# Boundless Generative Adversarial Network
This code is part of my project work in generating synthetic Ground Penetrating Radar Grams
using Generative Adversarial Networks. The model architecture is fully based on <a href="https://arxiv.org/abs/1908.07007"> Boundless </a>.

### Installation

The model requirements are given in the requirements.txt file.
The most important dependency is pytorch 1.7.1

Install the dependencies in your local environment.

```sh
$ pip install -r requirements.txt
```

### General Useage

Training the model

```sh
$ python model.py
```

Reset Standard Model

```sh
$ python reset.py
```

Generate panorama sample (experimental)
```sh
$ python panorama_sample.py
```

### Dataset
Create "/data" and place your own dataset in it. The subfolder structure is irrelevant.

