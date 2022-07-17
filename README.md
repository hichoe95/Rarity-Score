# Rarity Score

[Paper: Rarity Score : A New Metric to Evaluate the Uncommonness of Synthesized Images](https://arxiv.org/abs/2206.08549)

Jiyeon Han<sup>1</sup>, Hwanil Choi<sup>1</sup>, Yunjey Choi<sup>2</sup>, 
Junho Kim<sup>2</sup>, Jung-Woo Ha<sup>2</sup>, Jaesik Choi<sup>1</sup>    


<sup>1</sup> <sub>Graduate School of AI, KAIST</sub>
<sup>2</sup> <sub>NAVER AI Lab</sub>
<p align="center">
    <img src=sample.png width="900"> 
</p>

### Abstract
Evaluation metrics in image synthesis play a key role to measure performances of generative models. However, most metrics mainly focus on image fidelity. Existing diversity metrics are derived by comparing distributions, and thus they cannot quantify the diversity or rarity degree of each generated image. In this work, we propose a new evaluation metric, called `rarity score', to measure the individual rarity of each image synthesized by generative models. We first show empirical observation that common samples are close to each other and rare samples are far from each other in nearest-neighbor distances of feature space. We then use our metric to demonstrate that the extent to which different generative models produce rare images can be effectively compared. We also propose a method to compare rarities between datasets that share the same concept such as CelebA-HQ and FFHQ. Finally, we analyze the use of metrics in different designs of feature spaces to better understand the relationship between feature spaces and resulting sparse images. Code will be publicly available online for the research community.


### Rarity score
<p align="center">
    <img src=rarity_score.png width="900"> 
</p>
We hypothesize that ordinary samples would be closer to each other whereas unique and rare samples would be sparsely located in the feature space. 

For the fake feature <img src="https://latex.codecogs.com/png.image?\dpi{110}&space;\inline&space;\phi_g\in\mathbf{\Phi_g}" title="\inline \phi_g\in\mathbf{\Phi_g}" />and the real feature <img src="https://latex.codecogs.com/png.image?\dpi{110}&space;\inline&space;\phi_r\in\mathbf{\Phi_r}" title="\inline \phi_r\in\mathbf{\Phi_r}" />,our rarity score is defined as below

<img src="https://latex.codecogs.com/png.image?\dpi{150}&space;\text{Rarity}(\phi_g,\mathbf{\Phi_r})&space;=&space;\min_{r,&space;s.t.&space;\phi_g\in&space;B_k(\phi_r,\mathbf{\Phi_r})&space;}NN_k(\phi_r,\mathbf{\Phi_r})" title="\text{Rarity}(\phi_g,\mathbf{\Phi_r}) = \min_{r, s.t. \phi_g\in B_k(\phi_r,\mathbf{\Phi_r}) }NN_k(\phi_r,\mathbf{\Phi_r})" />

where <img src="https://latex.codecogs.com/png.image?\dpi{110}&space;\inline&space;B_k(\phi_i,\mathbf{\Phi})" title="\inline B_k(\phi_i,\mathbf{\Phi})" /> is the k-NN sphere of <img src="https://latex.codecogs.com/png.image?\dpi{110}&space;\inline&space;\phi_i" title="\inline \phi_i" /> with the radius of <img src="https://latex.codecogs.com/png.image?\dpi{100}&space;\inline&space;$NN_k(\phi_i,\mathbf{\Phi})$" title="\inline $NN_k(\phi_i,\mathbf{\Phi})$" />.

The proposed rarity score defines the rarity of an individual fake sample as the radius of k-NN sphere of a real sample which contains the given fake sample. By taking the minimum of the radii, we can prevent overestimating the sparsity of the given fake sample. As the fidelity of the samples outside of the real manifold is not guaranteed, our metric discards the generations which is not contained in the real manifold.

### Histograms of rarity score
The histogram of rarity score can be used to compare between the models or the datasets. 
For example, the histogram moves to the left as the truncation parameter gets smaller (the generations get similar to the average generation). 

<p align="center">
    <img src=truncation_histogram.png width="900"> 
</p>

Or, for the different datasets with the same concept, such as FFHQ and CelebA-HQ, the more diverse the dataset is, the higher the rarity scores are and the histogram moves to the right. 

<p align="center">
    <img src=dataset_histogram.png width="900"> 
</p>

### Example
```python
from rarity_score import *
import numpy as np

feature_dim = 2048
num_samples = 100
nearest_k = 3

real_features = np.random.normal(size=(num_samples, feature_dim))
fake_features = np.random.normal(size=(num_samples, feature_dim))


manifold = MANIFOLD(real_features=real_features, fake_features=fake_features)

score, score_index = manifold.rarity(k=nearest_k)

print(score[score_index])
```



### License
```
Copyright 2022-present NAVER Corp.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### Cite this work

```
@article{han2022rarity,
  title={Rarity Score: A New Metric to Evaluate the Uncommonness of Synthesized Images},
  author={Han, Jiyeon and Choi, Hwanil and Choi, Yunjey and Kim, Junho and Ha, Jung-Woo and Choi, Jaesik},
  journal={arXiv preprint arXiv:2206.08549},
  year={2022}
}
```
